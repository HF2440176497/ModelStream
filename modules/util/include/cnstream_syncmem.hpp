/*********************************************************************************************************
 * All modification made by Cambricon Corporation: © 2018--2019 Cambricon Corporation
 * All rights reserved.
 * All other contributions:
 * Copyright (c) 2014--2018, the respective contributors
 * All rights reserved.
 * For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Intel Corporation nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************************************************/

#ifndef CNSTREAM_SYNCMEM_HPP_
#define CNSTREAM_SYNCMEM_HPP_

/**
 * @file cnstream_syncmem.hpp
 *
 * This file contains a declaration of the CNSyncedMemory class.
 */

#include <cstddef>
#include <mutex>
#include <memory>
#include <map>

#include "cnstream_common.hpp"
#include "cnstream_logging.hpp"

namespace cnstream {

/**
 * @enum SyncedHead
 *
 * @brief An enumerator describing the synchronization status.
 */
enum class SyncedHead {
  UNINITIALIZED,  ///< The memory is not allocated.
  HEAD_AT_CPU,    ///< The data is updated to CPU but is not synchronized to CUDA yet.
  HEAD_AT_CUDA,  ///< The data is updated to CUDA but is not synchronized to CPU yet.
  HEAD_AT_NPU,  ///< The data is updated to NPU but is not synchronized to CPU yet.
  SYNCED          ///< The data is synchronized to both CPU and CUDA.
};

/**
 * @class CNSyncedMemory
 *
 * @brief CNSyncedMemory is a class synchronizing memory between CPU and Device.
 */
class CNSyncedMemory : private NonCopyable {
 public:
  explicit CNSyncedMemory(size_t size);
  explicit CNSyncedMemory(size_t size, int dev_id);
  virtual ~CNSyncedMemory();

 public:
  /**
   * @brief Gets the device ID.
   */
  int GetDevId() const;

  /**
   * @brief Sets the CPU data. 需要保证 data 的大小与 size_ 相等
   *
   * @param[in] data The data pointer on CPU.
   *
   * @return Void.
   */
  void SetCpuData(void* data);

  /**
   * @brief Gets the CPU data.
   *
   * @param No return value.
   *
   * @return Returns the CPU data pointer.
   *
   * @note If the size is 0, nullptr is always returned.
   */
  const void* GetCpuData();

  /**
   * @brief Gets the mutable CPU data.
   *
   * @return Returns the CPU data pointer.
   */
  void* GetMutableCpuData();

  /**
   * @brief Gets synchronization status.
   *
   * @return Returns synchronization status .
   *
   * @see SyncedHead.
   */
  SyncedHead GetHead() const { return head_; }
  /**
   * @brief Gets data bytes.
   *
   * @return Returns data bytes.
   */
  size_t GetSize() const { return size_; }

public:
  virtual void ToCpu();

#ifndef UNIT_TEST
 public:
#else
 private:
#endif
  /**
   * 通过 CNSyncedMemory 分配的，CNSyncedMemory 同时负责回收
   * Allocates memory by ``CNSyncedMemory`` if a certain condition is true.
   */
  std::unordered_map<DevType, bool> own_dev_data_;
  void* cpu_ptr_ = nullptr;
  SyncedHead head_ = SyncedHead::UNINITIALIZED;  ///< Identifies which device data is synchronized on.
  size_t size_ = 0;

  int dev_id_ = -1;
  mutable std::mutex mutex_;
};  // class CNSyncedMemory

}  // namespace cnstream

#endif  // CNSTREAM_SYNCMEM_HPP_
