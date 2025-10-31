#include <iostream>
#include "cnstream_module.hpp"

namespace cnstream {
class TestModuleOne : public Module, public ModuleCreator<TestModuleOne> {
public:
    explicit TestModuleOne(const std::string& name = "ModuleOne") : Module(name) {}
    bool Open(ModuleParamSet params) override {return true;}
    void Close() override {}
    int Process(std::shared_ptr<CNFrameInfo> frame_info) override {return 0;}
};

REGISTER_MODULE(TestModuleOne);

static void PrintAllModulesDesc() {
    std::vector<std::string> modules = ModuleFactory::Instance()->GetRegisted();
    std::cout << "--------- PrintAllModulesDesc: " << std::endl;
    std::cout << "modules: ";
    for (auto& it : modules) {
        std::cout << it << " ";
    }
    std::cout << std::endl;
}

}  // end cnstream


int main(int argc, char* argv[]) {
    cnstream::PrintAllModulesDesc();
    return 0;
}

