/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/


#include <memory>
#include <string>
#include <thread>
#include <map>

#include "cnstream_module.hpp"
#include "cnstream_pipeline.hpp"

// #include "profiler/pipeline_profiler.hpp"


namespace cnstream {

Module::~Module() {
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    container_->ReturnModuleIdx(id_);
  }
}

void Module::SetContainer(Pipeline* container) {
  if (container) {
    {
      RwLockWriteGuard guard(container_lock_);
      container_ = container;
    }
    GetId();
  } else {
    RwLockWriteGuard guard(container_lock_);
    container_ = nullptr;
    id_ = INVALID_MODULE_ID;
  }
}

size_t Module::GetId() {
  if (id_ == INVALID_MODULE_ID) {
    RwLockReadGuard guard(container_lock_);
    if (container_)
      id_ = container_->GetModuleIdx();
  }
  return id_;
}

bool Module::PostEvent(EventType type, const std::string& msg) {
  Event event;
  event.type = type;
  event.message = msg;
  event.module_name = name_;

  return PostEvent(event);
}

/**
 * 通过 Pipeline 的 EventBus 成员 PostEvent
 */
bool Module::PostEvent(Event e) {
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    return container_->GetEventBus()->PostEvent(e);
  } else {
    LOGW(CORE) << "[" << GetName() << "] module's container is not set";
    return false;
  }
}

int Module::DoTransmitData(std::shared_ptr<CNFrameInfo> data) {
  if (data->IsEos() && data->payload && IsStreamRemoved(data->stream_id)) {
    // FIMXE
    SetStreamRemoved(data->stream_id, false);
  }
  RwLockReadGuard guard(container_lock_);
  if (container_) {
    return container_->ProvideData(this, data);
  } else {
    if (HasTransmit()) NotifyObserver(data);  // 可以设置 Module 缺省 Observer
    return 0;
  }
}

/**
 * 仅在 Pipeline::TaskLoop 中调用
 */
int Module::DoProcess(std::shared_ptr<CNFrameInfo> data) {
  bool removed = IsStreamRemoved(data->stream_id);
  if (!removed) {
    // For the case that module is implemented by a pipeline
    if (data->payload && IsStreamRemoved(data->payload->stream_id)) {
      SetStreamRemoved(data->stream_id, true);
      removed = true;
    }
  }

  if (!HasTransmit()) {  // 默认通过 Pipeline 传输
    if (!data->IsEos()) {
      if (!removed) {
        int ret = Process(data);
        if (ret != 0) {
          return ret;
        }
      }
    } else {
      this->OnEos(data->stream_id);
    }
    return DoTransmitData(data);  // DoTransmitData 借助 Pipeline 实现传输
  } else {
    if (removed) {
      data->flags |= static_cast<size_t>(CNFrameFlag::CN_FRAME_FLAG_REMOVED);
    }
    return Process(data);  // Module 内的 Process 进行传输
  }
  return -1;
}

/**
 * 相比 DoProcess，TransmitData() 并没有调用 Process
 * 调用处：cnstream_source 和 infer_trans_data_helper
 */
bool Module::TransmitData(std::shared_ptr<CNFrameInfo> data) {
  if (!HasTransmit()) {
    return true;
  }
  // 必须要求 HasTransmit() == true
  if (!DoTransmitData(data)) {
    return true;
  }
  return false;
}

#ifndef CLOSE_PROFILER
ModuleProfiler* Module::GetProfiler() {
  RwLockReadGuard guard(container_lock_);
  if (container_ && container_->GetProfiler())
    return container_->GetProfiler()->GetModuleProfiler(GetName());
  return nullptr;
}
#endif

ModuleFactory* ModuleFactory::factory_ = nullptr;


class TestModuleOne : public Module, public ModuleCreator<TestModuleOne> {
  public:
   explicit TestModuleOne(const std::string& name = "ModuleOne") : Module(name) {}
   bool Open(ModuleParamSet params) override {return true;}
   void Close() override {}
   int Process(std::shared_ptr<CNFrameInfo> frame_info) override {return 0;}
 };


}  // namespace cnstream
