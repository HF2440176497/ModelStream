
#include "memop.hpp"
#include "libyuv/convert.h"
#include "libyuv/convert_argb.h"


namespace cnstream {

/**
 * @brief 在 Collection 中查找或者注册 buffer
 */
Buffer& MemoryBufferCollection::GetBuffer(DevType type, size_t size, int device_id = -1) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    if (it->second.size >= size) {
      return it->second;
    }
  }
  Buffer empty_buffer{nullptr, size, device_id};
  auto result = buffers_.emplace(type, std::move(empty_buffer));
  return result.first->second;
}

bool MemoryBufferCollection::Has(DevType type) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.find(type) != buffers_.end();
}

Buffer* MemoryBufferCollection::Get(DevType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    return &it->second;
  }
  return nullptr;
}

void MemoryBufferCollection::Clear(DevType type) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = buffers_.find(type);
  if (it != buffers_.end()) {
    buffers_.erase(it);
  }
}

void MemoryBufferCollection::ClearAll() {
  std::lock_guard<std::mutex> lock(mutex_);
  buffers_.clear();
}

size_t MemoryBufferCollection::GetDeviceCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return buffers_.size();
}

std::shared_ptr<void> MemOp::Allocate(size_t bytes)  {
  bytes = RoundUpSize(bytes);
  return cnCpuMemAlloc(bytes);
}

void MemOp::Copy(void* dst, const void* src, size_t size)  {
  memcpy(dst, src, size);
}

int MemOp::GetDeviceId() const { return -1; }

std::shared_ptr<CNSyncedMemory> MemOp::CreateSyncedMemory(size_t size) {
  return std::make_shared<CNSyncedMemory>(size);
}

void MemOp::SetData(std::shared_ptr<CNSyncedMemory> mem, void* data) {
  mem->SetCpuData(data);
}

/**
 * @brief 使用 CPU, 将解码帧转换为 dst_fmt 格式
 * @param dst 目标内存地址 由 memop->Allocate 分配
 * @note 目前仅支持到 RGB BGR 的转换
 */
int MemOp::ConvertImageFormat(void* dst, CNDataFormat dst_fmt,
                              const DecodeFrame* src_frame) {
  int width = src_frame->width;
  int height = src_frame->height;
  if (dst_fmt != CNDataFormat::CN_PIXEL_FORMAT_BGR24 &&
      dst_fmt != CNDataFormat::CN_PIXEL_FORMAT_RGB24) {
    LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
               << static_cast<int>(dst_fmt);
    return -1;
  }
  CNDataFormat src_fmt = src_frame->fmt;
  if (src_fmt == dst_fmt) {
    LOGW(CORE) << "MemOp::ConvertImageFormat: Source format is same as destination format";
    memcpy(dst, src_frame->plane[0], width * height * 3);
    return 0;
  }
  size_t dst_stride = width * 3; // 假设目标格式是RGB24或BGR24
  int ret = 0;

  switch (src_fmt) {
    case CNDataFormat::CN_PIXEL_FORMAT_BGR24: {
      if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_RGB24) {
        ret = libyuv::BGR24ToRGB24(
          static_cast<const uint8_t*>(src_frame->plane[0]), width * 3,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_BGR24) {
        ret = libyuv::BGR24ToBGR24(
          static_cast<const uint8_t*>(src_frame->plane[0]), width * 3,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source BGR24";
        return -1;
      }
      break;
    }
    case CNDataFormat::CN_PIXEL_FORMAT_RGB24: {
      if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_RGB24) {
        ret = libyuv::RGB24ToRGB24(
          static_cast<const uint8_t*>(src_frame->plane[0]), width * 3,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_BGR24) {
        ret = libyuv::RGB24ToBGR24(
          static_cast<const uint8_t*>(src_frame->plane[0]), width * 3,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source RGB24";
        return -1;
      }
      break;
    }
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV12: {
      if (src_frame->planeNum < 2) {
        LOGE(CORE) << "MemOp::ConvertImageFormat: NV12 format requires 2 planes";
        return -1;
      }
      const uint8_t* y_plane = static_cast<const uint8_t*>(src_frame->plane[0]);
      const uint8_t* uv_plane = static_cast<const uint8_t*>(src_frame->plane[1]);
      int y_stride = width;
      int uv_stride = width;
      if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_BGR24) {  // NV12 -> BGR24
        ret = libyuv::NV12ToBGR24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_RGB24) {  // NV12 -> RGB24
        ret = libyuv::NV12ToRGB24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV12";
        return -1;
      }
      break;
    }
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV21: {
      if (src_frame->planeNum < 2) {
        LOGE(CORE) << "MemOp::ConvertImageFormat: NV21 format requires 2 planes";
        return -1;
      }
      const uint8_t* y_plane = static_cast<const uint8_t*>(src_frame->plane[0]);
      const uint8_t* uv_plane = static_cast<const uint8_t*>(src_frame->plane[1]);
      int y_stride = width;
      int uv_stride = width;
      
      if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_BGR24) {  // NV21 -> BGR24
        ret = libyuv::NV21ToBGR24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else if (dst_fmt == CNDataFormat::CN_PIXEL_FORMAT_RGB24) {  // NV21 -> RGB24
        ret = libyuv::NV21ToRGB24(
          y_plane, y_stride,
          uv_plane, uv_stride,
          static_cast<uint8_t*>(dst), dst_stride,
          width, height);
      } else {
        LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV21";
        return -1;
      }
      break;
    }
    default:
      LOGE(CORE) << "MemOp::ConvertImageFormat: Unsupported source format " 
                 << static_cast<int>(src_fmt);
      return -1;
  }
  if (ret != 0) {
    LOGE(CORE) << "MemOp::ConvertImageFormat: libyuv conversion failed with error code: " << ret;
    return ret;
  }
  return 0;
}

}


