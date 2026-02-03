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

#include "cnstream_conveyor.hpp"

#include <chrono>
#include <memory>
#include <thread>
#include <vector>

namespace cnstream {

Conveyor::Conveyor(size_t max_size) : max_size_(max_size) {
}

uint32_t Conveyor::GetBufferSize() {
  std::unique_lock<std::mutex> lk(data_mutex_);
  return dataq_.size();
}

bool Conveyor::IsEmpty() {
  std::unique_lock<std::mutex> lk(data_mutex_);
  return dataq_.empty();
}

/**
 * 推送数据
 */
bool Conveyor::PushDataBuffer(FrameInfoPtr data, bool block) {
  std::unique_lock<std::mutex> lk(data_mutex_);
  
  // 阻塞模式等待队列有空间，非阻塞模式检查队列是否有空间
  // 阻塞模式下，最多等待 rel_time_ 但是 wait_for return true 说明一定满足队列空间要求
  if ((block && notfull_cond_.wait_for(lk, rel_time_, [this] { return dataq_.size() < max_size_; })) || 
      (!block && dataq_.size() < max_size_)) {
    dataq_.push(data);  // data: std::shared_ptr
    fail_time_ = 0;
    lk.unlock();
    notempty_cond_.notify_one();
    return true;
  }
  // 非阻塞模式或者阻塞模式超时后，队列仍满
  fail_time_ += 1;
  lk.unlock();
  return false;
}

uint64_t Conveyor::GetFailTime() {
  std::unique_lock<std::mutex> lk(data_mutex_);
  return fail_time_;
}

/**
 * 最多等待超时时间 rel_time_，如果超时还没有数据，返回 nullptr
 */
FrameInfoPtr Conveyor::PopDataBuffer() {
  std::unique_lock<std::mutex> lk(data_mutex_);
  FrameInfoPtr data = nullptr;
  if (notempty_cond_.wait_for(lk, rel_time_, [&] { return !dataq_.empty(); })) {
    data = dataq_.front();
    dataq_.pop();
    lk.unlock();
    notfull_cond_.notify_one();
    return data;
  }
  return data;
}

std::vector<FrameInfoPtr> Conveyor::PopAllDataBuffer() {
  std::unique_lock<std::mutex> lk(data_mutex_);
  std::vector<FrameInfoPtr> vec_data;
  FrameInfoPtr data = nullptr;
  while (!dataq_.empty()) {
    data = dataq_.front();
    dataq_.pop();
    vec_data.push_back(data);
  }
  return vec_data;
}

}  // namespace cnstream
