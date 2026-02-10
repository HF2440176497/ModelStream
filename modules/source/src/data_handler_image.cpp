
#include "cnstream_source.hpp"  // DataSource
#include "data_handler_image.hpp"

namespace cnstream {

std::shared_ptr<SourceHandler> ImageHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::make_shared<ImageHandler>(module, stream_id);
}

ImageHandler::ImageHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
  impl_ = std::make_unique<ImageHandlerImpl>(module, this);
}

ImageHandler::~ImageHandler() {
  Close();
}

bool ImageHandler::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: File handler open failed, no memory left";
    return false;
  }
  if (stream_index_ == INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

void ImageHandler::Close() {
  if (impl_) {
    impl_->Close();  // for image_impl: close consumer thread
    impl_.reset();
  }
}

void ImageHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

bool ImageHandlerImpl::Open() {
  param_ = module_->GetSourceParam();
  image_path_ = param_.file_path_;
  if (image_path_.empty() || access(image_path_.c_str(), F_OK) == -1) {
    LOGE(SOURCE) << "ImageHandlerImpl: Image path not found: " << image_path_;
    return false;
  }
  image_ = cv::imread(image_path_);
  if (image_.empty()) {
    LOGE(SOURCE) << "ImageHandlerImpl: Failed to load image: " << image_path_;
    return false;
  }
  running_.store(true);
  thread_ = std::thread(&ImageHandlerImpl::Loop, this);
  return true;
}

void ImageHandlerImpl::Stop() {
  if (running_.load()) {
    running_.store(false);
  }
}

void ImageHandlerImpl::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

/**
 * @brief 循环读取图片，模拟 decode 生成 DecodeFrame
 * 调用处：ImageHandlerImpl::Open
 */
void ImageHandlerImpl::Loop() {
  if (image_.empty()) {
    LOGE(SOURCE) << "ImageHandlerImpl: Failed to load image: " << image_path_;
    return;
  }
  FrController controller(framerate_);
  if (framerate_ > 0) controller.Start();

  // note: image_handler 直接手动指定 BGR24, 视频流解码时则需要 decoder 决定
  DecodeFrame frame(image_.rows, image_.cols, DataFormat::PIXEL_FORMAT_BGR24);
  frame.dev_type = DevType::CPU;
  frame.planeNum = 1;  // BGR格式使用1个平面
  
  // 分配内存并复制数据
  size_t data_size = image_.rows * image_.cols * 3;  // BGR格式每个像素3字节
  uint8_t* buffer = new (std::nothrow) uint8_t[data_size];
  if (!buffer) {
    LOGE(SOURCE) << "ImageHandlerImpl: Failed to allocate memory for image data";
    return;
  }
  memcpy(buffer, image_.data, data_size);
  frame.plane[0] = buffer;
  frame.stride[0] = image_.cols * 3;  // BGR格式每个像素3字节
  frame.buf_ref = std::make_unique<MatBufRef>(buffer);

  while (running_.load()) {
    controller.Control();
    frame.pts += 1000 / framerate_;
    std::shared_ptr<CNFrameInfo> data = OnDecodeFrame(&frame);
    if (!module_ || !handler_) {
      LOGE(SOURCE) << "ImageHandler: [" << stream_id_ << "]: module_ or handler_ is null";
      break;
    }
    handler_->SendData(data);
  }
  OnEndFrame();
}

/**
 * 定义如何处理来自数据源图像
 * 调用处：Loop 线程
 */
std::shared_ptr<CNFrameInfo> ImageHandlerImpl::OnDecodeFrame(DecodeFrame* frame) {
  if (!frame) {
    LOGW(SOURCE) << "[ImageHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return nullptr;
  }
  std::shared_ptr<CNFrameInfo> data = this->CreateFrameInfo();  // 规定了 CNFrameInfo 内含的基本基本成员
  if (!data) {
    LOGW(SOURCE) << "[ImageHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return nullptr;
  }
  data->timestamp = frame->pts;
  if (!frame->valid) {
    data->flags = static_cast<size_t>(CNFrameFlag::CN_FRAME_FLAG_INVALID);
    this->SendFrameInfo(data);
    return nullptr;
  }
  int ret = SourceRender::Process(data, frame, frame_id_++, param_);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: SetupDataFrame function, failed to setup data frame.";
    return nullptr;
  }
  return data;
}

/**
 * Handler 线程循环结束时, 发送结束帧
 */
void ImageHandlerImpl::OnEndFrame() {
  // 调用 SourceRender::OnEndFrame 发送 EOS 帧
  std::shared_ptr<CNFrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGW(SOURCE) << "[FileHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return;
  }
  this->SendFrameInfo(data);
  LOGI(SOURCE) << "[ImageHandlerImpl] OnEndFrame function, send end frame.";
}


}  // namespace cnstream
