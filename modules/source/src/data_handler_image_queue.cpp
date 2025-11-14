

#include "data_source.hpp"

#include "private/cnstream_constants_pri.hpp"
#include "data_handler_image_queue.hpp"

#include "data_handler_util.hpp"


namespace cnstream {

/**
 * static method 调用构造函数
 */
std::shared_ptr<SourceHandler> ImageQueueHandler::Create(DataSource *module, const std::string &stream_id) {
  if (!module) {
    LOGE(SOURCE) << "[" << stream_id << "]: module_ null";
    return nullptr;
  }
  return std::make_shared<ImageQueueHandler>(module, stream_id);
}

ImageQueueHandler::ImageQueueHandler(DataSource *module, const std::string &stream_id)
    : SourceHandler(module, stream_id) {
  impl_ = std::make_unique<ImageQueueHandlerImpl>(module, this);
}

ImageQueueHandler::~ImageQueueHandler() {
  Close();
}

bool ImageQueueHandler::Open() {
  if (!module_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
    return false;
  }
  if (!impl_) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: File handler open failed, no memory left";
    return false;
  }
  if (stream_index_ == cnstream::INVALID_STREAM_IDX) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: Invalid stream_idx";
    return false;
  }
  return impl_->Open();
}

void ImageQueueHandler::Close() {
  if (impl_) {
    impl_->Close();  // for image_queue_impl: close consumer thread
    impl_.reset();
  }
}

void ImageQueueHandler::Stop() {
  if (impl_) {
    impl_->Stop();
  }
}

bool ImageQueueHandler::IsRunning() const { 
  return running_.load(); 
}

void ImageQueueHandler::PushDatas(std::vector<uint64_t> timestamps, std::vector<cv::Mat> images) {
  if (timestamps.size() != images.size()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: timestamps size not equal to images size";
    return;
  }
  for (int i = 0; i < timestamps.size(); i++) {
    std::shared_ptr<ImageFrame> frame = std::make_shared<ImageFrame>();
    frame->valid = true;
    frame->pts = timestamps[i];
    frame->width = images[i].cols;
    frame->height = images[i].rows;
    frame->fmt = ImageFrame::PixFmt::FMT_BGR;
    frame->data = images[i].clone();  // 深拷贝图像数据
    frame->device_id = -1;
    frame->planeNum = 1;
    queue_.Push(frame);
  }
  return;
}

bool ImageQueueHandlerImpl::Open() {
  DataSource *source = dynamic_cast<DataSource *>(module_);
  param_ = source->GetSourceParam();

  running_.store(true);
  thread_ = std::thread(&ImageQueueHandlerImpl::Loop, this);
  return true;
}

void ImageQueueHandlerImpl::Stop() {
  if (running_.load()) {
    running_.store(false);
  }
}

void ImageQueueHandlerImpl::Close() {
  Stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

/**
 * 作为消费者线程
 */
void ImageQueueHandlerImpl::Loop() {
  int frame_rate = 20;
  while (!running_.load()) {
    std::shared_ptr<ImageFrame> frame;
    if (!queue_.WaitAndTryPop(frame, std::chrono::milliseconds(50))) {
      continue;
    }
    OnDecodeFrame(frame);
    if (!module_) {
      LOGE(SOURCE) << "[" << stream_id_ << "]: module_ null";
      break;
    }
  }
  OnEndFrame();
}

/**
 * 定义如何处理来自数据源图像
 * 为了和 CNStream 保持一致 仍称 OnDecodeFrame
 * 调用处：Loop 线程
 */
void ImageQueueHandlerImpl::OnDecodeFrame(std::shared_ptr<ImageFrame> frame) {
  if (frame_count_++ % param_.interval_ != 0) {
    return;  // discard frames
  }
  if (!frame) {
    LOGW(SOURCE) << "[FileHandlerImpl] OnDecodeFrame function frame is nullptr.";
    return;
  }
  std::shared_ptr<CNFrameInfo> data = this->CreateFrameInfo();  // 规定了 CNFrameInfo 内含的基本基本成员
  if (!data) {
    LOGW(SOURCE) << "[FileHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return;
  }
  data->timestamp = frame->pts;  // FIXME
  if (!frame->valid) {
    data->flags = static_cast<size_t>(CNFrameFlag::CN_FRAME_FLAG_INVALID);
    this->SendFrameInfo(data);
    return;
  }
  // TODO: 这里的 SetupDataFrame 暂定为抽象接口
  int ret = SetupDataFrame(data, frame, frame_id_++, param_);
  if (ret < 0) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: SetupDataFrame function, failed to setup data frame.";
    return;
  }
  this->SendFrameInfoConn(data);
}


/**
 * Sasha: 为了借助 Pipeline 进行数据传输管理，放入自己的 connector
 */
void ImageQueueHandlerImpl::SendFrameInfoConn(std::shared_ptr<CNFrameInfo> data) {
  if (!module_->GetConnector()) {
    LOGE(SOURCE) << "[" << stream_id_ << "]: connector not connected";
    return;
  }
  FrController controller(frame_rate);
  controller.Start();
  int conveyor_idx = stream_id_ % (module_->GetConnector()->GetConveyorCount());
  module_->GetConnector()->PushDataBufferToConveyor(conveyor_idx, data);
  controller.Control();
}

/**
 * Handler 线程循环结束时, 发送结束帧
 */
void ImageQueueHandlerImpl::OnEndFrame() {
  // 调用 SourceRender::OnEndFrame 发送 EOS 帧
  std::shared_ptr<CNFrameInfo> data = this->CreateFrameInfo(true);
  if (!data) {
    LOGW(SOURCE) << "[FileHandlerImpl] OnDecodeFrame function, failed to create FrameInfo.";
    return;
  }
  this->SendFrameInfo(data);
  LOGI(SOURCE) << "[ImageQueueHandlerImpl] OnEndFrame function, send end frame.";
}

int ImageQueueHandlerImpl::SetupDataFrame(std::shared_ptr<CNFrameInfo> frame_info,
                                          std::shared_ptr<ImageFrame> frame, uint64_t frame_id,
                                          const DataSourceParam &param_) {
  if (!frame_info || !frame) return -1;
  if (!frame->valid) {  // 不应该出现 所以直接返回
    LOGW(SOURCE) << "[ImageQueueHandlerImpl] SetupDataFrame function, "
                 << "frame is invalid.";
    return -1;
  }
  CNDataFramePtr dataframe = frame_info->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
  if (!dataframe) return -1;

  dataframe->frame_id = frame_id;
  /*fill source data info*/
  dataframe->width = frame->width;
  dataframe->height = frame->height;

  // 目前只是支持 BGR24
  dataframe->fmt = CNDataFormat::CN_PIXEL_FORMAT_BGR24;  // same as cv::Mat
  if (dataframe->GetPlanes() != 1) {
    LOGE(SOURCE) << "[ImageQueueHandlerImpl] SetupDataFrame function, "
                 << "only support single plane image.";
    return -1;
  }

  // convert to cpu first always
  dataframe->ctx.dev_type = DevContext::DevType::CPU;
  dataframe->ctx.dev_id = -1;
  dataframe->ctx.ddr_channel = -1;  // unused for cpu

  // 临时测试，直接将 cv::Mat 数据赋值，后面 dataframe 直接通过 GetBGR() 方法获取
  dataframe->bgr_data = frame->data;
  return 0;
}


}  // namespace cnstream