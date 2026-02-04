
/**
* 测试 Json 文件读取，配置解析
*/

#include "base.hpp"
#include "cnstream_config.hpp"

static std::string test_pipeline_json = "pipeline.json";

TEST(JSON, ReadFile) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    rapidjson::Document doc;
    doc.Parse(json_str.c_str());
    EXPECT_FALSE(doc.HasParseError()) << "Parse json file failed";

    EXPECT_TRUE(doc.HasMember("profiler_config")) << "Json file has no profiler_config field";
    EXPECT_TRUE(doc.HasMember("decoder")) << "Json file has no decoder field";
    EXPECT_TRUE(doc.HasMember("sort_h")) << "Json file has no sort field";
    EXPECT_TRUE(doc.HasMember("osd")) << "Json file has no osd field";
}

/**
* 测试 Json 文件读取后，读取某字段再得到字符串
* @detail 测试 Inferencer 字段
*/
TEST(JSON, ReadFile2Str) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    rapidjson::Document doc;
    doc.Parse(json_str.c_str());
    EXPECT_FALSE(doc.HasParseError()) << "Parse json file failed";

    std::string infer_name = "InferencerYolo";
    EXPECT_TRUE(doc.HasMember(infer_name.c_str())) << "Json file has no Inferencer field";
    const rapidjson::Value& inferencer = doc[infer_name.c_str()];
    EXPECT_TRUE(inferencer.IsObject()) << "Inferencer field is not object";
    
    rapidjson::StringBuffer buffer;
    rapidjson::Writer writer(buffer);
    inferencer.Accept(writer);
    std::string inferencer_str(buffer.GetString());
    EXPECT_TRUE(!inferencer_str.empty()) << "Inferencer field is empty";
    LOGI(COREUNITEST) << "Inferencer field: " << inferencer_str << std::endl;
}

/**
 * @brief 创建一个临时文件，测试 CNConfigBase 基类
 */
TEST(CoreConfig, ParseByJSONFile) {
    struct TestConfig : public cnstream::CNConfigBase {
        bool ParseByJSONStr(const std::string& jstr) override {return true;}
    };
    TestConfig test_config;
    auto config_file = CreateTempFile("pipeline_temp");  // fd-filename
    EXPECT_TRUE(test_config.ParseByJSONStr(config_file.second));
    EXPECT_TRUE(test_config.config_root_dir.empty());
    unlink(config_file.second.c_str());
    close(config_file.first);  // close fd
}

/**
 * @brief 测试 CNModuleConfig 解析，将字段 Inferencer 解析为 CNModuleConfig
 */
TEST(CoreConfig, ModuleConfig) {
    std::string json_str = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_str.empty()) << "Read json file failed";
    rapidjson::Document doc;
    doc.Parse(json_str.c_str());
    EXPECT_FALSE(doc.HasParseError()) << "Parse json file failed";

    std::string infer_name = "InferencerYolo";
    EXPECT_TRUE(doc.HasMember(infer_name.c_str())) << "Json file has no Inferencer field";
    const rapidjson::Value& inferencer = doc[infer_name.c_str()];
    EXPECT_TRUE(inferencer.IsObject()) << "Inferencer field is not object";
    
    rapidjson::StringBuffer buffer;
    rapidjson::Writer writer(buffer);
    inferencer.Accept(writer);
    std::string inferencer_str(buffer.GetString());

    // CMoudleConfig
    cnstream::CNModuleConfig inferencer_config;
    inferencer_config.config_root_dir = "./";
    EXPECT_TRUE(inferencer_config.ParseByJSONStr(inferencer_str));
    EXPECT_TRUE(inferencer_config.name.empty());
    EXPECT_EQ(inferencer_config.className, "cnstream::Inferencer");
    EXPECT_EQ(inferencer_config.next.size(), 1);  // next_modules 

    LOGI(COREUNITEST) << "Inferencer next modules: ";
    for (const auto& elem : inferencer_config.next) {
        LOGI(COREUNITEST) << "module: " << elem << " ";
    }
    EXPECT_EQ(inferencer_config.config_root_dir, inferencer_config.parameters[CNS_JSON_DIR_PARAM_NAME]);
}

/**
 * @brief Test CNGraphConfig 
 * @detail 这是最大一级的解析层级
 */
TEST(CoreConfig, CNGraphConfig) {
    std::string json_content = readFile(test_pipeline_json.c_str());
    EXPECT_FALSE(json_content.empty()) << "Read json file failed";

    cnstream::CNGraphConfig graph_config;
    graph_config.config_root_dir = "./";
    EXPECT_TRUE(graph_config.ParseByJSONStr(json_content));

    // 检查 profiler_config 模块
    EXPECT_FALSE(graph_config.profiler_config.enable_profile);
}
