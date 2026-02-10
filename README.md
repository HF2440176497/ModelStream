# ModelStream

ModelStream 是一个高性能的视频结构化分析框架，采用模块化设计，支持灵活的流水线配置和并行处理。该框架参考了 CNStream 的设计理念，提供了完整的视频数据处理、推理和结果输出能力。

## 特性

- **模块化架构**: 基于流水线的模块化设计，易于扩展和定制
- **灵活配置**: 支持 JSON 配置文件，可动态构建处理流水线
- **并行处理**: 支持多线程并行处理，提高处理效率
- **数据流管理**: 完善的数据流管理和事件机制
- **跨平台**: 支持 Linux 平台，可选 NVIDIA CUDA 支持

## 项目结构

```
ModelStream/
├── framework/              # 核心框架
│   └── core/
│       ├── include/       # 核心头文件
│       │   ├── cnstream_pipeline.hpp    # 流水线管理
│       │   ├── cnstream_module.hpp      # 模块基类
│       │   ├── cnstream_frame.hpp       # 帧信息
│       │   ├── cnstream_config.hpp      # 配置管理
│       │   ├── cnstream_eventbus.hpp    # 事件总线
│       │   ├── cnstream_connector.hpp   # 模块连接器
│       │   └── ...
│       └── src/           # 核心源文件
├── modules/               # 功能模块
│   ├── source/            # 数据源模块
│   │   └── include/
│   │       └── data_source.hpp
│   ├── util/              # 工具模块
│   │   ├── include/
│   │   └── src/
│   └── unittest/          # 模块单元测试
├── 3rdparty/              # 第三方库
│   ├── libyuv/            # YUV 转换库
│   ├── googletest/        # 单元测试框架
│   └── lohmann-json/      # JSON 解析库
├── cmake/                 # CMake 配置
├── samples/               # 示例程序
├── tools/                 # 工具程序
└── CMakeLists.txt         # 主构建文件
```

## 核心组件

### Pipeline (流水线)

Pipeline 是整个框架的管理器，负责：
- 管理模块的创建和连接
- 控制数据在模块间的传输
- 处理事件和消息传递
- 管理流水线的生命周期（启动/停止）

### Module (模块)

Module 是所有功能模块的基类，提供：
- 统一的模块接口
- 参数配置和校验
- 数据处理接口
- 事件发布机制

### SourceModule (数据源模块)

数据源模块的基类，用于：
- 从各种数据源读取数据（视频文件、图像、RTSP 流等）
- 将数据注入到流水线中

### FrameInfo (帧信息)

帧信息类，包含：
- 流标识符 (stream_id)
- 时间戳 (timestamp)
- 数据集合 (Collection)
- 帧状态标志

### Connector (连接器)

模块间数据传输的桥梁，支持：
- 多队列并发传输
- 流量控制
- 数据缓冲

## 依赖项

### 必需依赖

- **gflags** - 命令行参数解析
- **glog** - 日志库
- **OpenCV** - 图像处理库
- **nlohmann/json** - JSON 解析（已包含在 3rdparty 中）

### 可选依赖

- **libyuv** - YUV 格式转换（可内置编译）
- **googletest** - 单元测试（可内置编译）
- **NVIDIA CUDA** - GPU 加速支持

## 编译安装

### 1. 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get install cmake build-essential
sudo apt-get install libgflags-dev libgoogle-glog-dev
sudo apt-get install libopencv-dev
```

### 2. 编译项目

```bash
# 创建构建目录
mkdir build && cd build

# 配置构建选项
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MODULES=ON \
  -DBUILD_TESTS=ON \
  -DBUILD_LIBYUV=ON

# 编译
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 3. 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `build_modules` | ON | 构建功能模块 |
| `build_modules_contrib` | OFF | 构建额外模块 |
| `build_samples` | OFF | 构建示例程序 |
| `build_tests` | OFF | 构建单元测试 |
| `build_libyuv` | ON | 编译 libyuv 库 |
| `build_tools` | OFF | 构建工具程序 |
| `build_python_api` | OFF | 构建 Python API |
| `NVIDIA_PLATFORM` | OFF | 启用 NVIDIA CUDA 支持 |

## 使用示例

### 1. JSON 配置文件

创建一个配置文件 `config.json`:

```json
{
  "profiler_config": {
    "enable_profile": true
  },
  "datasource": {
    "parallelism": 2,
    "max_input_queue_size": 20,
    "class_name": "cnstream::DataSource",
    "next_modules": ["inference"],
    "custom_params": {
      "source_type": "image",
      "input_path": "./images"
    }
  },
  "inference": {
    "parallelism": 4,
    "max_input_queue_size": 30,
    "class_name": "cnstream::Inference",
    "next_modules": [],
    "custom_params": {
      "model_path": "./models/detect.mlu",
      "label_path": "./models/labels.txt"
    }
  }
}
```

### 2. C++ 代码示例

```cpp
#include "cnstream_pipeline.hpp"
#include "cnstream_module.hpp"

int main() {
    // 创建流水线
    cnstream::Pipeline pipeline("my_pipeline");

    // 从 JSON 文件构建流水线
    if (!pipeline.BuildPipelineByJSONFile("config.json")) {
        std::cerr << "Failed to build pipeline" << std::endl;
        return -1;
    }

    // 启动流水线
    if (!pipeline.Start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return -1;
    }

    // 等待处理完成
    // ...

    // 停止流水线
    pipeline.Stop();

    return 0;
}
```

### 3. 自定义模块

```cpp
#include "cnstream_module.hpp"

class MyModule : public cnstream::Module, public cnstream::ModuleCreator<MyModule> {
 public:
  explicit MyModule(const std::string& name) : cnstream::Module(name) {}

  bool Open(cnstream::ModuleParamSet param_set) override {
    // 初始化模块资源
    return true;
  }

  void Close() override {
    // 释放模块资源
  }

  int Process(std::shared_ptr<cnstream::FrameInfo> data) override {
    // 处理数据
    // 返回 0 表示成功，数据将自动传递到下一模块
    // 返回 >0 表示已处理，需要手动调用 TransmitData
    // 返回 <0 表示错误
    return 0;
  }
};

REGISTER_MODULE(MyModule);
```

## 已完成功能

### 框架核心
- [x] Pipeline 流水线管理
- [x] Module 模块基类
- [x] SourceModule 数据源基类
- [x] FrameInfo 帧信息管理
- [x] Connector 模块连接器
- [x] EventBus 事件总线
- [x] Graph 图结构管理
- [x] JSON 配置解析

### 功能模块
- [x] DataSource 数据源模块（支持图像输入）
- [x] Util 工具模块（内存管理、分配器等）

### 工具和测试
- [x] 单元测试框架集成
- [x] Profiler 性能分析工具

## 待开发功能

### 推理模块
- [ ] InferenceModule 通用推理模块
- [ ] 支持多种推理引擎（MLU、CUDA、TensorRT 等）
- [ ] 模型加载和管理
- [ ] 批处理优化

### 数据消费模块
- [ ] EncoderModule 编码模块
- [ ] DecoderModule 解码模块
- [ ] RenderModule 渲染模块
- [ ] SinkModule 结果输出模块

### 数据源扩展
- [ ] VideoSource 视频文件源
- [ ] RTSPSource RTSP 流源
- [ ] CameraSource 摄像头源
- [ ] DirectorySource 目录批量处理

### 其他功能
- [ ] Python 绑定
- [ ] 更多示例程序
- [ ] 性能优化
- [ ] 文档完善

## 性能特性

- **零拷贝传输**: 通过智能指针管理，减少数据拷贝
- **并行处理**: 支持多模块并行执行
- **流水线优化**: 异步数据传输，提高吞吐量
- **内存池**: 高效的内存分配和回收

## 致谢

本项目参考了 [CNStream](https://github.com/Cambricon/CNStream) 的设计理念。
