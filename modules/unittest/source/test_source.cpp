
#include "base.hpp"

#include "data_handler_image_queue.hpp"
#include "data_source.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include <chrono>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

namespace cnstream {


uint64_t getCurrentTimestampMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

class EosObserver: public StreamMsgObserver {
  public:
    void Update(const StreamMsg &msg) override {
    }
};

class SourceModuleTest : public testing::Test {
  protected:
    virtual void SetUp() {
      std::string json_content = readFile(test_pipeline_json.c_str());
      EXPECT_FALSE(json_content.empty()) << "Read json file failed";
      cnstream::CNGraphConfig graph_config;
      graph_config.config_root_dir = "./";
      graph_config.ParseByJSONStr(json_content);
      graph_config_ = graph_config;

      std::string class_name = "cnstream::DataSource";  // Factory 的 map key
      std::string init_name = "DataSource";
      ResetParam(test_param_);
      module_.reset(ModuleFactory::Instance()->Create(class_name, init_name));
      image_handler_.reset(ImageQueueHandler::Create(module_.get(), stream_id_));
    }
    virtual void TearDown() {  // 当前用例结束
      LOGI(SourceModuleTest) << "TearDown";
    }
  protected:
    void StartPushDataThread(int loop_count = 20) {
        push_data_thread = std::thread([this, loop_count]() {
            this->PushDataWorker(loop_count);
        });
    }
    void StopPushDataThread() {
        if (push_data_thread.joinable()) {
            push_data_thread.join();
        }
    }
    void PushDataWorker(int loop_count) {
      if (!image_handler_) {
        std::cout << "image_handler_ is null, push data failed" << std::endl;
        return;
      }
      std::string image_path = "../data/images/1.jpg";
      cv::Mat image = cv::imread(image_path);
      std::vector<ImageFrame> image_frames {image};
      while (loop_count--) {
          uint64_t timestamp = getCurrentTimestampMs();
          image_handler_->PushDatas({timestamp}, image_frames);
          std::this_thread::sleep_for(std::chrono::milliseconds(100));  // FPS 约为 10
      }
    }
  protected:
    const std::string stream_id_ = "stream_0";
    std::thread push_data_thread;
    std::shared_ptr<SourceHandler> image_handler_ = nullptr;
    std::shared_ptr<Module> module_ = nullptr;
    cnstream::CNGraphConfig graph_config_;
    ModuleParamSet test_param_;
};


void ResetParam(ModuleParamSet &param) {
  param["output_type"] = "cpu";
  param["device_id"] = "0";
  param["interval"] = "1";
  param["decoder_type"] = "cpu";
}

TEST_F(SourceModuleTest, AddSource) {
  EXPECT_NE(module_, nullptr);
  EXPECT_TRUE(module_->CheckParamSet(test_param_));
  EXPECT_TRUE(module_->Open(test_param_));
  DataSource *source = std::dynamic_cast<DataSource*>(module_.get());
  EXPECT_NE(source, nullptr);
  EXPECT_EQ(source->AddSource(image_handler_), 0);
  EXPECT_EQ(source->RemoveSources(), 0);  // 移除所有的 source
}

/**
 * 读取图片
 */
TEST_F(SourceModuleTest, Loop) {
  Pipeline pipeline("test_pipeline");
  pipeline.BuildPipeline(graph_config_);  // 此处假设 Pipeline 测试过，直接构建

  EXPECT_NE(module_, nullptr);
  EXPECT_EQ(module_->Open(test_param_), true);
  DataSource *source = std::dynamic_cast<DataSource*>(module_.get());
  EXPECT_NE(source, nullptr);
  source->AddSource(image_handler_);

  // 开启 Pipeline
  EXPECT_TRUE(pipeline.Start());
  EXPECT_FALSE(IsStreamRemoved(stream_id_));  // 此处应该没有被移除
  
  // 手动发送 EOS 此时 sourcemodule 是不能移除 handler 的
  image_handler_->impl_->SendFlowEos();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 等待 EOS 消息被处理

  // 同步等待
  LOGI(SourceModuleTest) << "Wait for EOS message to be processed";
  LOGI(SourceModuleTest) << "CheckStreamEosReached(stream_id_) = " << CheckStreamEosReached(stream_id_, true);
  LOGI(SourceModuleTest) << "Wait for EOS message complete";

  source->RemoveSources();  // 移除所有的 source
  EXPECT_FALSE(IsStreamRemoved(stream_id_));  // 此处应该被移除标志，所以 not find
}

/**
 * 此用例检查运行时状态 connector 是否正常 线程运行
 */
TEST_F(SourceModuleTest, ThreadRun) {
  Pipeline pipeline("test_pipeline");
  pipeline.BuildPipeline(graph_config_);  // 此处假设 Pipeline 测试过，直接构建

  EXPECT_NE(module_, nullptr);
  EXPECT_EQ(module_->Open(test_param_), true);
  DataSource *source = std::dynamic_cast<DataSource*>(module_.get());
  EXPECT_NE(source, nullptr);
  source->AddSource(image_handler_);

  EXPECT_TRUE(pipeline.Start());
  StartPushDataThread();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 等待数据被处理
  StopPushDataThread();

  // 需要检查的线程: event_bus connector 线程是否运行
  EXPECT_FALSE(pipeline.GetEventBus()->IsStopped());
  auto connector = source->GetConnector(stream_id_);
  EXPECT_NE(connector, nullptr);
  EXPECT_TRUE(connector->IsRunning());

  source->RemoveSources();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  EXPECT_FALSE(connector->IsRunning());
  EXPECT_FALSE(image_handler_->impl_->IsRunning());
}


}  // namespace cnstream