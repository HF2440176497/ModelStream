# ModelStream 数据流水线线程安全分析报告

## 1. 概述

本报告分析 ModelStream 框架中数据从 SourceModule 产生到 Pipeline 流水线传输过程中的线程安全问题。

---

## 2. 线程模型

### 2.1 线程类型

| 线程类型 | 创建位置 | 职责 |
|---------|---------|------|
| **主线程** | 测试程序 | 执行测试代码、控制 Pipeline 生命周期 |
| **Loop 线程** | `ImageHandlerImpl::Open()` | 产生帧数据，调用 `OnDecodeFrame` |
| **TaskLoop 线程** | `Pipeline::Start()` | 每个 Module 有 `parallelism` 个处理线程 |
| **EventBus 线程** | `EventBus::Start()` | 处理事件消息 |
| **StreamMsg 线程** | `Pipeline::Start()` | 处理流消息 |

### 2.2 线程数量

```
总线程数 = 1(主线程) + 1(Loop线程) + Σ(parallelism) + 1(EventBus) + 1(StreamMsg)
```

---

## 3. 数据流分析

### 3.1 完整数据流路径

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Loop 线程                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ImageHandlerImpl::Loop()                                                    │
│       │                                                                       │
│       ├── 创建 DecodeFrame (栈对象)                                           │
│       ├── memcpy(buffer, image_.data, data_size)  ← image_ 只在此线程访问    │
│       ├── OnDecodeFrame(&frame)                                              │
│       │       │                                                               │
│       │       ├── CreateFrameInfo() → FrameInfo::Create()                   │
│       │       ├── SourceRender::Process()                                    │
│       │       │       ├── 设置 frame->width/height/stride                    │
│       │       │       ├── buf_ref 所有权转移 → deAllocator_                   │
│       │       │       └── CopyToSyncMem() → cv::cvtColor()                   │
│       │       └── return FrameInfo (shared_ptr)                              │
│       │                                                                       │
│       └── handler_->SendData(data)                                          │
│               │                                                               │
│               └── SourceModule::SendData(data)                              │
│                       │                                                       │
│                       └── DoTransmitData(data)                              │
│                               │                                               │
└───────────────────────────────┼─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Pipeline::ProvideData()                               │
│                        (Loop 线程调用)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Conveyor::PushDataBufferToConveyor()                                 │    │
│  │                                                                       │    │
│  │  std::unique_lock<std::mutex> lk(data_mutex_);  ← 线程安全            │    │
│  │  dataq_.push(data);  // shared_ptr 引用计数+1                        │    │
│  │  notempty_cond_.notify_one();  // 唤醒等待的 TaskLoop                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TaskLoop 线程                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  Pipeline::TaskLoop()                                                        │
│       │                                                                       │
│       ├── connector->PopDataBufferFromConveyor()                            │
│       │       │                                                               │
│       │       └── Conveyor::PopDataBuffer()                                 │
│       │               std::unique_lock<std::mutex> lk(data_mutex_);          │
│       │               data = dataq_.front();                                 │
│       │               dataq_.pop();                                          │
│       │                                                                       │
│       ├── module->DoProcess(data)                                           │
│       │       │                                                               │
│       │       ├── Process(data)  // 用户自定义处理                           │
│       │       └── DoTransmitData(data) → 传递给下一模块                      │
│       │                                                                       │
│       └── 循环处理下一帧                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 共享资源分析

### 4.1 SourceModule::source_map_

```cpp
// cnstream_source.hpp
std::mutex mutex_;
std::map<std::string, std::shared_ptr<SourceHandler>> source_map_;
```

**访问场景**：
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| `AddSource()` | 主线程 | `std::unique_lock<std::mutex>` | ✅ 安全 |
| `RemoveSource()` | 主线程 | `std::unique_lock<std::mutex>` | ✅ 安全 |
| `GetSourceHandler()` | 任意线程 | `std::unique_lock<std::mutex>` | ✅ 安全 |

**结论**：所有访问都有 `mutex_` 保护，线程安全。

---

### 4.2 Conveyor 数据队列

```cpp
// cnstream_conveyor.hpp
std::queue<FrameInfoPtr> dataq_;
std::mutex data_mutex_;
std::condition_variable notempty_cond_;
std::condition_variable notfull_cond_;
```

**访问场景**：
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| `PushDataBuffer()` | Loop 线程 | `std::unique_lock<std::mutex>` | ✅ 安全 |
| `PopDataBuffer()` | TaskLoop 线程 | `std::unique_lock<std::mutex>` | ✅ 安全 |

**结论**：使用 mutex + condition_variable 实现生产者-消费者模型，线程安全。

---

### 4.3 全局流状态变量

```cpp
// cnstream_common_pri.hpp
inline std::mutex s_eos_lock_;
inline std::map<std::string, std::atomic<bool>> s_stream_eos_map_;

inline std::mutex s_remove_lock_;
inline std::set<std::string> s_stream_removed_set_;
```

**访问场景**：

#### s_stream_eos_map_
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| 插入 (EOS帧创建) | Loop 线程 | `s_eos_lock_` | ✅ 安全 |
| 更新 (EOS帧析构) | TaskLoop 线程 | `s_eos_lock_` | ✅ 安全 |
| 查询 (CheckStreamEosReached) | 主线程 | `s_eos_lock_` | ✅ 安全 |

#### s_stream_removed_set_
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| 插入 (RemoveSource) | 主线程 | `s_remove_lock_` | ✅ 安全 |
| 删除 (RemoveSource完成) | 主线程 | `s_remove_lock_` | ✅ 安全 |
| 查询 (IsStreamRemoved) | Loop/TaskLoop | `s_remove_lock_` | ✅ 安全 |

**结论**：所有访问都有对应的 mutex 保护，线程安全。

---

### 4.4 FrameInfo::modules_mask_

```cpp
// cnstream_frame.hpp
mutable std::mutex mask_lock_;
uint64_t modules_mask_ = 0;
```

**访问场景**：
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| `SetModulesMask()` | TaskLoop (Pipeline::ProvideData) | `mask_lock_` | ✅ 安全 |
| `GetModulesMask()` | TaskLoop | `mask_lock_` | ✅ 安全 |
| `MarkPassed()` | TaskLoop | `mask_lock_` | ✅ 安全 |

**结论**：有 mutex 保护，线程安全。

---

### 4.5 DataFrame 成员变量

```cpp
// cnstream_frame_va.hpp
int width;
int height;
int stride[CN_MAX_PLANES];
std::unique_ptr<CNSyncedMemory> data[CN_MAX_PLANES];
std::mutex mtx;
cv::Mat mat_;
```

**访问场景**：
| 操作 | 线程 | 锁保护 | 安全性 |
|------|------|--------|--------|
| 初始化 (CopyToSyncMem) | Loop 线程 | 无锁 | ⚠️ 单线程写入 |
| `GetImage()` | TaskLoop 线程 | `mtx` | ✅ 安全 |
| `HasImage()` | TaskLoop 线程 | `mtx` | ✅ 安全 |

**关键分析**：
- `DataFrame` 在 Loop 线程中创建和初始化
- 初始化完成后，通过 `shared_ptr` 传递给 TaskLoop
- **写入和读取不在同一时间段**，不存在数据竞争

**结论**：通过 `shared_ptr` 的所有权转移机制，确保写入完成后再读取，线程安全。

---

## 5. 潜在问题分析

### 5.1 已发现问题：stride 未初始化

**问题描述**：
```cpp
// data_handler_util.cpp (修复前)
frame->width = decode_frame->width;
frame->height = decode_frame->height;
// frame->stride 未被设置！
```

**影响**：
- `GetPlaneBytes()` 使用未初始化的 `stride[0]`
- 导致内存分配大小错误
- `cv::cvtColor` 访问非法内存

**状态**：已修复

---

### 5.2 潜在问题：Loop 线程与 Pipeline 停止的竞争

**场景**：
```
主线程                          Loop线程                    TaskLoop线程
   │                               │                            │
   │── SendFlowEos()               │                            │
   │                               │── 继续创建帧 ──────────────►│
   │                               │                            │
   │── 测试结束，析构              │                            │
   │   image_handler_ 析构         │                            │
   │   image_ 被释放               │── memcpy(image_.data) ❌   │
   │                               │      访问已释放内存         │
```

**解决方案**：
1. 先调用 `Stop()` 停止 Loop 线程
2. 调用 `Close()` 等待线程结束
3. 最后销毁相关资源

---

### 5.3 潜在问题：EOS 处理时序

**场景**：
```
Loop线程                         TaskLoop线程                  主线程
   │                               │                            │
   │── SendFlowEos()               │                            │
   │   创建 EOS FrameInfo          │                            │
   │   s_stream_eos_map_[id]=false │                            │
   │                               │                            │
   │                               │── 处理 EOS 帧              │
   │                               │   FrameInfo 析构           │
   │                               │   s_stream_eos_map_[id]=true│
   │                               │                            │
   │                               │                    CheckStreamEosReached()
   │                               │                    检查 EOS 状态
```

**分析**：
- EOS 帧创建时设置 `false`
- EOS 帧析构时设置 `true`
- `CheckStreamEosReached()` 同步等待 `true`

**结论**：时序正确，但需要注意：
- 同步等待会阻塞主线程
- 如果 TaskLoop 处理缓慢，主线程会长时间阻塞

---

## 6. 内存管理分析

### 6.1 帧数据生命周期

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Loop 线程创建                                                            │
│    DecodeFrame frame;                          // 栈对象                    │
│    uint8_t* buffer = new uint8_t[data_size];   // 堆内存                    │
│    frame.buf_ref = make_unique<MatBufRef>(buffer);                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. 所有权转移 (SourceRender::Process)                                       │
│    frame->deAllocator_ = make_unique<Deallocator>(                          │
│        decode_frame->buf_ref.release());                                    │
│    // buffer 的所有权从 DecodeFrame 转移到 DataFrame                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. DataFrame 持有 buffer                                                    │
│    DataFrame::deAllocator_ 持有 MatBufRef                                   │
│    当 DataFrame 析构时，MatBufRef::~MatBufRef() 执行 delete[] buffer        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 shared_ptr 引用计数

```
FrameInfo (shared_ptr)
    │
    ├── Loop 线程持有 (创建时)
    │       引用计数 = 1
    │
    ├── Push 到 Conveyor
    │       引用计数 = 2 (Loop线程 + 队列)
    │
    ├── Loop 线程释放局部变量
    │       引用计数 = 1 (队列)
    │
    ├── TaskLoop Pop
    │       引用计数 = 2 (队列 + TaskLoop)
    │
    ├── 队列 Pop 完成
    │       引用计数 = 1 (TaskLoop)
    │
    └── TaskLoop 处理完成
            引用计数 = 0 → 析构
```

**结论**：`shared_ptr` 的引用计数机制确保内存在最后一个引用释放后才被回收，线程安全。

---

## 7. 总结

### 7.1 线程安全评估

| 组件 | 线程安全机制 | 评估 |
|------|-------------|------|
| SourceModule::source_map_ | mutex | ✅ 安全 |
| Conveyor | mutex + condition_variable | ✅ 安全 |
| s_stream_eos_map_ | mutex | ✅ 安全 |
| s_stream_removed_set_ | mutex | ✅ 安全 |
| FrameInfo::modules_mask_ | mutex | ✅ 安全 |
| DataFrame | shared_ptr 所有权转移 | ✅ 安全 |
| ImageHandlerImpl::image_ | 单线程访问 | ✅ 安全 |

### 7.2 关键发现

1. **stride 未初始化问题**：已修复，这是导致崩溃的直接原因

2. **Loop 线程生命周期管理**：
   - 必须在销毁 `image_handler_` 之前停止 Loop 线程
   - 正确顺序：`Stop()` → `Close()` → 销毁

3. **数据所有权转移清晰**：
   - `DecodeFrame` → `DataFrame` 的 buffer 所有权转移
   - `shared_ptr<FrameInfo>` 的引用计数管理

### 7.3 最佳实践建议

1. **测试代码中**：
   ```cpp
   // 正确的清理顺序
   image_handler_->impl_->Stop();    // 1. 停止 Loop 线程
   image_handler_->impl_->Close();   // 2. 等待线程结束
   pipeline_->Stop();                // 3. 停止 Pipeline
   image_handler_.reset();           // 4. 释放 handler
   ```

2. **Pipeline 停止时**：
   - 先关闭 SourceModule（触发 Loop 线程停止）
   - 再停止 Connector 和 TaskLoop 线程
   - 最后清理资源

---

## 8. 附录：关键代码位置

| 功能 | 文件路径 |
|------|---------|
| ImageHandler Loop | `modules/source/src/data_handler_image.cpp:119` |
| SourceRender::Process | `modules/source/src/data_handler_util.cpp:9` |
| Conveyor 线程安全队列 | `framework/core/src/cnstream_conveyor.cpp` |
| Pipeline::ProvideData | `framework/core/src/cnstream_pipeline.cpp:460` |
| Pipeline::TaskLoop | `framework/core/src/cnstream_pipeline.cpp:524` |
| 全局流状态管理 | `framework/core/src/private/cnstream_common_pri.cpp` |

---

*报告生成时间：2026-02-13*
