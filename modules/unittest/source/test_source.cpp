
#include "base.hpp"

#include "cnstream_logging.hpp"
#include "data_handler_image.hpp"
#include "data_source.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

#include <chrono>
#include <typeinfo>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

namespace cnstream {

static std::string test_pipeline_json = "pipeline_base_test.json";
static std::vector<std::string> expected_nodes = {"DataSource", "InferencerVoid"};

// 在测试实例中，定义出这个 virtual module
class InferencerVoid: public Module, public ModuleCreator<InferencerVoid> {
  public:
    InferencerVoid(const std::string &name) : Module(name) {}
    ~InferencerVoid() {}
    bool Open(ModuleParamSet params) override {
      return true;
    }
    void Close() override {
      LOGI(InferencerVoid) << "Close";
    }
    int Process(std::shared_ptr<CNFrameInfo> frame) override {
      LOGI(InferencerVoid) << "Process frame " << frame->stream_id << "; with time: " << frame->timestamp;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return 0;
    }
};

REGISTER_MODULE(InferencerVoid);

static uint64_t getCurrentTimestampMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

static void ResetParam(ModuleParamSet &param) {
  param["output_type"] = "cpu";
  param["device_id"] = "0";
  param["interval"] = "1";
  param["decoder_type"] = "cpu";
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

      ResetParam(test_param_);

      pipeline_ = std::make_shared<Pipeline>("test_pipeline");
      EXPECT_NE(pipeline_, nullptr);
      EXPECT_TRUE(pipeline_->BuildPipeline(graph_config_));
    }

    virtual void TearDown() {  // 当前用例结束
      LOGI(SourceModuleTest) << "TearDown";
    }

  protected:
    const std::string stream_id_ = "channel-1";
    std::shared_ptr<ImageHandler> image_handler_ = nullptr;
    std::shared_ptr<DataSource> module_ = nullptr;
    std::shared_ptr<Pipeline> pipeline_ = nullptr;
    cnstream::CNGraphConfig graph_config_;
    ModuleParamSet test_param_;
};


TEST(Source, BasicOutput) {
  std::cout << "=== Standard output test ===" << std::endl;
  std::cout << "Calling LOGI..." << std::endl;
  LOGI(MODULESUNITEST) << "LOGI test message";
  std::cout << "LOGI call completed" << std::endl;
  LOGW(MODULESUNITEST) << "LOGW test message";
  LOGE(MODULESUNITEST) << "LOGE test message";
}

TEST_F(SourceModuleTest, PipelineInit) {

  // 先测试配置加载
  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config_));

  // 检查 Module 相关 mask 标志位
  // PS: 必须要在 Build 完成的 Pipeline 中看到
  std::cout << "----- Module Mask: " << std::endl;
  for (auto node_iter = pipeline_->graph_->DFSBegin(); node_iter != pipeline_->graph_->DFSEnd(); ++node_iter) {
    std::cout << "node name: " << node_iter->data.module->GetName() << std::endl;
    std::cout << "module id: " << node_iter->data.module->GetId() << std::endl;
    std::cout << "route_mask: " << node_iter->data.route_mask << std::endl;
    std::cout << "parent_nodes_mask: " << node_iter->data.parent_nodes_mask << std::endl;
  }
  // DataSource 标记 route_mask 非 0, parent_nodes_mask 为 0
  // InferencerVoid 标记 route_mask 为 0 (因为是头节点) parent_nodes_mask 非 0 

  // 发现：DataSource 的 route_mask 也包含了自身 Module 的标记

  // 应当含有 DataSource 和 InferencerVoid 
  ModuleFactory::Instance()->PrintRegistedModules();

}  // PipelineInit

/**
 * 读取图片
 */
TEST_F(SourceModuleTest, Loop) {
  // 提取 pipeline 中的 DataSource 模块
  EXPECT_NE(pipeline_, nullptr);
  Module* module_in_pipeline = pipeline_->GetModule("decoder");
  EXPECT_NE(module_in_pipeline, nullptr);

  ResetParam(test_param_);
  EXPECT_TRUE(module_in_pipeline->CheckParamSet(test_param_));
  EXPECT_TRUE(module_in_pipeline->Open(test_param_));
  DataSource *source = dynamic_cast<DataSource*>(module_in_pipeline);
  EXPECT_NE(source, nullptr);

  std::shared_ptr<SourceHandler> source_handler_ptr = ImageHandler::Create(source, stream_id_);
  image_handler_ = std::dynamic_pointer_cast<ImageHandler>(source_handler_ptr);
  EXPECT_NE(image_handler_, nullptr);

  // 开启 Pipeline
  EXPECT_TRUE(pipeline_->Start());
  EXPECT_FALSE(IsStreamRemoved(stream_id_));  // 此处应该没有被移除

  EXPECT_EQ(source->AddSource(image_handler_), 0);
  EXPECT_TRUE(image_handler_->impl_->running_);

  LOGI(SourceModuleTest) << "Handler stream idx: " << image_handler_->GetStreamIndex();
  EXPECT_NE(image_handler_->GetStreamIndex(), INVALID_STREAM_IDX);  // 等同 data->GetStreamIndex
  
  // 手动发送 EOS 此时 sourcemodule 是不能移除 handler 的
  EXPECT_TRUE(pipeline_->IsRunning());
  
  image_handler_->impl_->SendFlowEos();
  PrintStreamEos();  // 创建 eos 帧之后应当看到 eos_map 注册了 false
  EXPECT_FALSE(StreamEosMapValue(stream_id_));
  
  std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 等待 EOS 消息被处理

  // 同步等待
  LOGI(SourceModuleTest) << "Wait for EOS message to be processed";
  LOGI(SourceModuleTest) << "CheckStreamEosReached(stream_id_) = " << std::boolalpha << CheckStreamEosReached(stream_id_, true);
  LOGI(SourceModuleTest) << "Wait for EOS message complete";
}


// TODO: 我们如果启动多个线程，操作 Handler 发送数据


}  // namespace cnstream