

#include <mutex>
#include <memory>

#include "cnstream_frame_va.hpp"
#include "memop_factory.hpp"

namespace cnstream {


static
cv::Mat BGRToBGR(const DataFrame& frame) {
  const cv::Mat bgr(frame.height, frame.stride[0], CV_8UC3, const_cast<void*>(frame.data[0]->GetCpuData()));
  return bgr(cv::Rect(0, 0, frame.width, frame.height)).clone();
}

static
cv::Mat RGBToBGR(const DataFrame& frame) {
  const cv::Mat rgb(frame.height, frame.stride[0], CV_8UC3, const_cast<void*>(frame.data[0]->GetCpuData()));
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr(cv::Rect(0, 0, frame.width, frame.height)).clone();
}

/**
 * @brief 转换 NV12 或者 NV21 的 YUV420SP 图像到 BGR 格式
 */
static
cv::Mat YUV420SPToBGR(const DataFrame& frame, bool nv21) {
  const uint8_t* y_plane = reinterpret_cast<const uint8_t*>(frame.data[0]->GetCpuData());
  const uint8_t* uv_plane = reinterpret_cast<const uint8_t*>(frame.data[1]->GetCpuData());
  int width = frame.width;
  int height = frame.height;
  if (width <= 0 || height <= 1) {
    LOGF(FRAME) << "Invalid width or height, width = " << width << ", height = " << height;
  }
  height = height & (~static_cast<int>(1));

  int y_stride = frame.stride[0];
  int uv_stride = frame.stride[1];
  cv::Mat bgr(height, width, CV_8UC3);
  uint8_t* dst_bgr24 = bgr.data;
  int dst_stride = width * 3;
  if (nv21)
    libyuv::NV21ToRGB24Matrix(y_plane, y_stride, uv_plane, uv_stride,
                              dst_bgr24, dst_stride, &libyuv::kYvuH709Constants, width, height);
  else
    libyuv::NV12ToRGB24Matrix(y_plane, y_stride, uv_plane, uv_stride,
                              dst_bgr24, dst_stride, &libyuv::kYvuH709Constants, width, height);
  return bgr;
}

static inline
cv::Mat NV12ToBGR(const DataFrame& frame) {
  return YUV420SPToBGR(frame, false);
}

static inline
cv::Mat NV21ToBGR(const DataFrame& frame) {
  return YUV420SPToBGR(frame, true);
}

static inline
cv::Mat FrameToImageBGR(const DataFrame& frame) {
  switch (frame.fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
      return BGRToBGR(frame);
    case DataFormat::PIXEL_FORMAT_RGB24:
      return RGBToBGR(frame);
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
      return NV12ToBGR(frame);
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      return NV21ToBGR(frame);
    default:
      LOGF(FRAME) << "Unsupported pixel format. fmt[" << static_cast<int>(frame.fmt) << "]";
  }
  // never be here
  return cv::Mat();
}

/**
 * @brief 转换数据到 BGR 格式
 * 在数据存在于 CPU 上，才可调用
 */
cv::Mat DataFrame::GetImage() {
  std::lock_guard<std::mutex> lk(mtx);
  if (!mat_.empty()) {
    return mat_;
  }
  mat_ = FrameToImageBGR(*this);
  return mat_;
}

size_t DataFrame::GetPlaneBytes(int plane_idx) const {
  if (plane_idx < 0 || plane_idx >= GetPlanes()) return 0;
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return height * stride[0] * 3;
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      if (0 == plane_idx)
        return height * stride[0];
      else if (1 == plane_idx)
        return std::ceil(1.0 * height * stride[1] / 2);
      else
        LOGF(FRAME) << "plane index wrong.";
    default:
      return 0;
  }
  return 0;
}

size_t DataFrame::GetBytes() const {
  size_t bytes = 0;
  for (int i = 0; i < GetPlanes(); ++i) {
    bytes += GetPlaneBytes(i);
  }
  return bytes;
}

/**
 * @brief 每次调用查找已注册的 MemOp 创建器，根据当前 dev_type 和 dev_id 创建 MemOp
 * 调用处：CopyToSyncMem(decode_frame)
 */
std::unique_ptr<MemOp> DataFrame::CreateMemOp() {
  auto dev_type = this->ctx.dev_type;  // current dev and id
  int dev_id = this->ctx.dev_id;
  std::unique_ptr<MemOp> memop = MemOpFactory::Instance().CreateMemOp(dev_type, dev_id);
  if (!memop) {
    LOGF(FRAME) << "CreateMemOp: failed to create MemOp from " << static_cast<int>(dev_type) << " with dev_id " << dev_id;
    return nullptr;
  }
  return memop;
}

void DataFrame::CopyToSyncMem(DecodeFrame* decode_frame) {
  if (this->ctx.dev_type == DevType::INVALID) {
    LOGF(FRAME) << "CopyToSyncMem: dev_type is INVALID";
    return;
  }
  if (DataFormat::PIXEL_FORMAT_RGB24 != this->fmt) {
    LOGF(FRAME) << "CopyToSyncMem: fmt not RGB24, decode_frame fmt is " << static_cast<int>(decode_frame->fmt) << ", this fmt is " << static_cast<int>(this->fmt);
    return;
  }

  std::unique_ptr<MemOp> memop = CreateMemOp();  // 局部变量，只在当前次解码使用
  if (!memop) return;

  // 1. 零拷贝 直接复用内存
  if (this->deAllocator_ != nullptr && decode_frame->fmt == this->fmt) {
    for (int i = 0; i < GetPlanes(); i++) {
      const size_t plane_bytes = GetPlaneBytes(i);
      this->data[i] = memop->CreateSyncedMemory(plane_bytes);
      memop->SetData(this->data[i].get(), decode_frame->plane[i]);
    }
    return;
  }
  // 2. 分配内存-格式转换-创建SyncedMemory
  size_t bytes = GetBytes();
  bytes = ROUND_UP(bytes, 64 * 1024);
  
  std::shared_ptr<void> dst_buffer = memop->Allocate(bytes);
  void* dst_plane = dst_buffer.get();

  // 格式转换：分平面将转换后的数据写入 dst_plane
  if (decode_frame->fmt != this->fmt) {
    int ret = memop->ConvertImageFormat(dst_plane, this->fmt, decode_frame);
    if (ret != 0) {
      LOGF(FRAME) << "CopyToSyncMem: Format conversion failed with error code: " << ret;
      return;
    }
  }
  // 分平面将地址封装到 this->data，但 data 并不管理其生命周期，而是交给 mem_manager 管理
  for (int i = 0; i < GetPlanes(); ++i) {
    const size_t plane_bytes = GetPlaneBytes(i);
    this->data[i] = memop->CreateSyncedMemory(plane_bytes);
    memop->SetData(this->data[i].get(), dst_plane);
    dst_plane = static_cast<uint8_t*>(dst_plane) + plane_bytes;
  }
  Buffer& buffer = mem_manager_.GetBuffer(ctx.dev_type, bytes, ctx.dev_id);
  buffer.data = std::move(dst_buffer);  // buffer.data 会自动管理生命周期
  this->deAllocator_.reset();  // 非零拷贝场景，确保释放
}

// bool CNInferObject::AddAttribute(const std::string& key, const CNInferAttr& value) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (attributes_.find(key) != attributes_.end()) return false;

//   attributes_.insert(std::make_pair(key, value));
//   return true;
// }

// bool CNInferObject::AddAttribute(const std::pair<std::string, CNInferAttr>& attribute) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (attributes_.find(attribute.first) != attributes_.end()) return false;

//   attributes_.insert(attribute);
//   return true;
// }

// CNInferAttr CNInferObject::GetAttribute(const std::string& key) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (attributes_.find(key) != attributes_.end()) return attributes_[key];

//   return CNInferAttr();
// }

// bool CNInferObject::AddExtraAttribute(const std::string& key, const std::string& value) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (extra_attributes_.find(key) != extra_attributes_.end()) return false;

//   extra_attributes_.insert(std::make_pair(key, value));
//   return true;
// }

// bool CNInferObject::AddExtraAttributes(const std::vector<std::pair<std::string, std::string>>& attributes) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   bool ret = true;
//   for (auto& attribute : attributes) {
//     ret &= AddExtraAttribute(attribute.first, attribute.second);
//   }
//   return ret;
// }

// std::string CNInferObject::GetExtraAttribute(const std::string& key) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (extra_attributes_.find(key) != extra_attributes_.end()) {
//     return extra_attributes_[key];
//   }
//   return "";
// }

// bool CNInferObject::RemoveExtraAttribute(const std::string& key) {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   if (extra_attributes_.find(key) != extra_attributes_.end()) {
//     extra_attributes_.erase(key);
//   }
//   return true;
// }

// StringPairs CNInferObject::GetExtraAttributes() {
//   std::lock_guard<std::mutex> lk(attribute_mutex_);
//   return StringPairs(extra_attributes_.begin(), extra_attributes_.end());
// }


}  // namespace cnstream
