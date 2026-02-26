
#ifndef CNSTREAM_SYNCMEM_CUDA_HPP_
#define CNSTREAM_SYNCMEM_CUDA_HPP_

#include <cstddef>
#include <mutex>
#include <sstream>
#include <string>

#include "cnstream_common.hpp"
#include "cnstream_logging.hpp"

#include "cnstream_syncmem.hpp"
#include "cuda/cuda_check.hpp"

namespace cnstream {

class CNSyncedMemoryCuda : public CNSyncedMemory {
public:
  struct StatusInfoCuda {
    size_t size = 0;
    int dev_id = -1;
    SyncedHead head = SyncedHead::UNINITIALIZED;
    bool own_cpu_data = false;
    bool own_cuda_data = false;
  };

  CNSyncedMemoryCuda(size_t size);
  CNSyncedMemoryCuda(size_t size, int dev_id);
  ~CNSyncedMemoryCuda() override;

public:
  void ToCpu() override;

public:
  void ToCuda();
  void SetCudaData(void* data);
  const void* GetCudaData();
  void* GetMutableCudaData();

  StatusInfoCuda GetStatusInfoCuda() const {
    StatusInfoCuda info;
    info.size = GetSize();
    info.dev_id = GetDevId();
    info.head = GetHead();
    info.own_cpu_data = (cpu_ptr_ != nullptr);
    info.own_cuda_data = (cuda_ptr_ != nullptr);
    return info;
  }

  std::string StatusToString() const {
    StatusInfoCuda info = GetStatusInfoCuda();
    std::string head_str;
    switch (info.head) {
      case SyncedHead::UNINITIALIZED: head_str = "UNINITIALIZED"; break;
      case SyncedHead::HEAD_AT_CPU: head_str = "HEAD_AT_CPU"; break;
      case SyncedHead::HEAD_AT_CUDA: head_str = "HEAD_AT_CUDA"; break;
      case SyncedHead::HEAD_AT_NPU: head_str = "HEAD_AT_NPU"; break;
      case SyncedHead::SYNCED: head_str = "SYNCED"; break;
      default: head_str = "UNKNOWN"; break;
    }
    std::ostringstream oss;
    oss << "CNSyncedMemoryCuda{size=" << info.size
        << ", dev_id=" << info.dev_id
        << ", head=" << head_str
        << ", own_cpu_data=" << std::boolalpha << info.own_cpu_data
        << ", own_cuda_data=" << info.own_cuda_data << "}";
    return oss.str();
  }

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  void* cuda_ptr_ = nullptr;  ///< CUDA data pointer.
};

}  // namespace cnstream

#endif  // CNSTREAM_SYNCMEM_CUDA_HPP_