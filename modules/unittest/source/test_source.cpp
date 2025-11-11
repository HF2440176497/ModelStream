
#include "base.hpp"

#include "data_handler_image_queue.hpp"
#include "data_source.hpp"

namespace cnstream {

class EosObserver: public StreamMsgObserver {
  public:
    void Update(const StreamMsg &msg) override {
      if (msg.type == StreamMsgType::EOS_MSG) {
        SetStreamRemoved(msg.stream_id);
        LOGI(EosObserver) << "Received EOS message from module " << msg.module_name << " of stream " << msg.stream_id;
      }
    }
};

class SourceModuleTest : public testing::Test {
  protected:
    virtual void SetUp() {
      std::string class_name = "cnstream::DataSource";  // Factory 的 map key
      std::string init_name = "DataSource";
      ResetParam(test_param_);
      module_.reset(ModuleFactory::Instance()->Create(class_name, init_name));
    }
    virtual void TearDown() {  // 当前用例结束
      LOGI(SourceModuleTest) << "TearDown";
    }
  protected:
    std::shared_ptr<Module> module_ = nullptr;
    ModuleParamSet test_param_;
};

void ResetParam(ModuleParamSet &param) {  // NOLINT
  param["output_type"] = "cpu";
  param["device_id"] = "0";
  param["interval"] = "1";
  param["decoder_type"] = "cpu";
}

TEST(DataSource, AddSource) {
  EXPECT_NE(module_, nullptr);
  EXPECT_EQ(module_->Open(test_param_), true);
  DataSource *source = std::dynamic_cast<DataSource*>(module_.get());
  EXPECT_NE(source, nullptr);
  std::string stream_id = "stream_0";
  auto image_handler = ImageQueueHandler::Create(module_.get(), stream_id);
  EXPECT_EQ(source->AddSource(image_handler), 0);
  EXPECT_EQ(source->RemoveSources(), 0);  // 移除所有的 source
}

/**
 * 作为 Module 在 Pipeline 中，Start 后运行
 */
TEST(DataSource, Open) {

  
}

}  // namespace cnstream