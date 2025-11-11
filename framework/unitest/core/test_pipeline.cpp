

#include "base.hpp"
#include "cnstream_pipeline.hpp"


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
    bool Close() override {
      return true;
    }
    int Process(std::shared_ptr<CNFrameInfo> frame) override {
      LOGI(InferencerVoid) << "Process frame " << frame->frame_id << "; with time: " << frame->timestamp;
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      return 0;
    }
};

REGISTER_MODULE(InferencerVoid);


class PipelineConfigLoad : public testing::Test {
  protected:
    virtual void SetUp() {
      std::string json_content = readFile(test_pipeline_json.c_str());
      EXPECT_FALSE(json_content.empty()) << "Read json file failed";
      cnstream::CNGraphConfig graph_config;
      graph_config.config_root_dir = "./";
      graph_config.ParseByJSONStr(json_content);
      graph_config_ = graph_config;
    }
    virtual void TearDown() {  // 当前用例结束
      LOGI(TestConfigLoad) << "TearDown";
    }
  protected:
    cnstream::CNGraphConfig graph_config_;
};


TEST_F(PipelineConfigLoad, PipelineInit) {
  std::unique_ptr<CNGraph<NodeContext>> graph = std::make_unique<CNGraph<NodeContext>>();
  EXPECT_NE(nullptr, graph.get());
  EXPECT_TRUE(graph->Init(graph_config_));

  Pipeline pipeline("test_pipeline");
  

  EXPECT_TRUE(pipeline.graph_->Init(graph_config_));
  std::vector<std::shared_ptr<Module>> modules;
  EXPECT_TRUE(pipeline.CreateModules(modules));
  EXPECT_EQ(modules.size(), expected_nodes.size());

  pipeline.GenerateModulesMask();
  
  // 检查生成的 ID
  for (auto cur_node = pipeline.graph_->DFSBegin(); cur_node != pipeline.graph_->DFSEnd(); ++cur_node) {
    std::cout << "----- Node: "<< GetName() << "; Module id: " << cur_node->data.module->GetId() 
    << "; parent_nodes_mask: " << cur_node->data.parent_nodes_mask
    << std::endl
  }


  EXPECT_TRUE(pipeline.BuildPipeline(graph_config_));



  

}  // PipelineInit


}  // namespace cnstream