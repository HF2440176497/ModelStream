
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
#include <map>

#include "cnstream_frame.hpp"
#include "cnstream_module.hpp"

namespace cnstream {


std::shared_ptr<CNFrameInfo> CNFrameInfo::Create(const std::string& stream_id, bool eos,
                                                 std::shared_ptr<CNFrameInfo> payload) {
  if (stream_id == "") {
    LOGE(CORE) << "CNFrameInfo::Create() stream_id is empty string.";
    return nullptr;
  }
  std::shared_ptr<CNFrameInfo> ptr(new (std::nothrow) CNFrameInfo());
  if (!ptr) {
    LOGE(CORE) << "CNFrameInfo::Create() new CNFrameInfo failed.";
    return nullptr;
  }
  ptr->stream_id = stream_id;
  ptr->payload = payload;
  if (eos) {
    ptr->flags |= static_cast<size_t>(cnstream::CNFrameFlag::CN_FRAME_FLAG_EOS);
    if (!ptr->payload) {  // 不存在 payload 的情况下，查询字典中的 stream_id 需要准备好处理 EOS 
      std::lock_guard<std::mutex> guard(s_eos_lock_);
      s_stream_eos_map_[stream_id] = false;
    }
    return ptr;
  }
  return ptr;
}

CNS_IGNORE_DEPRECATED_PUSH
CNFrameInfo::~CNFrameInfo() {
  if (this->IsEos()) {
    if (!this->payload) {  // 不存在 payload 的情况下
      std::lock_guard<std::mutex> guard(s_eos_lock_);
      s_stream_eos_map_[stream_id] = true;
    }
    return;
  }
}
CNS_IGNORE_DEPRECATED_POP

void CNFrameInfo::SetModulesMask(uint64_t mask) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  modules_mask_ = mask;
}

uint64_t CNFrameInfo::GetModulesMask() {
  std::lock_guard<std::mutex> lk(mask_lock_);
  return modules_mask_;
}

uint64_t CNFrameInfo::MarkPassed(Module* module) {
  std::lock_guard<std::mutex> lk(mask_lock_);
  modules_mask_ |= (uint64_t)1 << module->GetId();
  return modules_mask_;
}

}  // namespace cnstream