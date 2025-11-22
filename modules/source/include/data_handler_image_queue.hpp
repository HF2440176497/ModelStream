

/**
 * 提供生产者-消费者的接口，读取图片
 */


#ifndef MODULES_SOURCE_HANDLER_IMAGE_QUEUE_HPP_
#define MODULES_SOURCE_HANDLER_IMAGE_QUEUE_HPP_

#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "cnstream_logging.hpp"
#include "data_source.hpp"
#include "data_handler_util.hpp"

namespace cnstream {

/**
 * 描述用于生成 CNFrameInfo 的图像结构
 */
struct ImageFrame {
  bool valid;
  uint64_t pts;  // timestamp in ms
  uint32_t width;
  uint32_t height;
  cv::Mat data;  // 存储图像数据
  enum class PixFmt { 
    FMT_INVALID, 
    FMT_NV12, 
    FMT_NV21, 
    FMT_I420, 
    FMT_YUYV,
    FMT_J420,
    FMT_BGR,
    FMT_RGB,
  } fmt;
  int32_t device_id;  // -1: CPU, 0: CUDA or other devices
  int32_t planeNum;
 public:
  ~ImageFrame() {}
};

class ImageQueueHandlerImpl: public SourceRender {
 public:
  explicit ImageQueueHandlerImpl(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}
  
  /**
   * Impl 接口通过 SourceHandler 调用
   */
  bool Open();
  void Close();
  void Stop();
  void Loop();

public:
  void OnEndFrame();
  std::shared_ptr<CNFrameInfo> OnDecodeFrame(std::shared_ptr<ImageFrame> frame);

private:
  int SetupDataFrame(std::shared_ptr<CNFrameInfo> frame_info,
                    std::shared_ptr<ImageFrame> frame, uint64_t frame_id,
                    const DataSourceParam &param_);
#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::atomic<bool> running_{false};
  ThreadSafeQueue<std::shared_ptr<ImageFrame>> queue_;  // 通过智能指针管理，需要得到 frame_info
  std::thread thread_;  // consumer thread
  DataSource *module_;
  std::string stream_id_;
  DataSourceParam param_;  // from DataSource
};

}  // namespace cnstream

#endif
