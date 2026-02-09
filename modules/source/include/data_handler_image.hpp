
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
#include "video_decoder.hpp"


namespace cnstream {

class ImageHandlerImpl: public SourceRender {
  struct MatBufRef : public IDecBufRef {
    explicit MatBufRef(void* data) : data_(data) {}
    ~MatBufRef() override {
      delete[] static_cast<uint8_t*>(data_);
    }
    void* data_;
  };
 public:
  explicit ImageHandlerImpl(DataSource *module, SourceHandler *handler)
      : SourceRender(handler), module_(module), stream_id_(handler->GetStreamId()) {}
  
  // Impl 接口通过 SourceHandler 调用
  bool Open();
  void Close();
  void Stop();
  void Loop();

public:
  void OnEndFrame();
  std::shared_ptr<CNFrameInfo> OnDecodeFrame(DecodeFrame* frame);

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::atomic<bool> running_{false};
  int framerate_ = 5;
  std::string image_path_;
  cv::Mat image_;
  std::thread thread_;  // consumer thread
  DataSource *module_;
  std::string stream_id_;
  DataSourceParam param_;  // from DataSource
};

}  // namespace cnstream

#endif
