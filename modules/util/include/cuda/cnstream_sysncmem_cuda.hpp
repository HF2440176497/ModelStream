
#ifndef CNSTREAM_SYNCMEM_CUDA_HPP_
#define CNSTREAM_SYNCMEM_CUDA_HPP_

#include <cstddef>
#include <mutex>

#include "cnstream_common.hpp"
#include "cnstream_logging.hpp"

#include "cnstream_sysncmem.hpp"
#include "cuda/cuda_check.hpp"

namespace cnstream {

class CNSyncedMemoryCuda : public CNSyncedMemory {
public:
  CNSyncedMemoryCuda(size_t size) : CNSyncedMemory(size) {}
  CNSyncedMemoryCuda(size_t size, int dev_id) : CNSyncedMemory(size, dev_id) {}
  ~CNSyncedMemoryCuda() override;

public:
  void ToCpu() override;

public:
  void ToCuda();
  void SetCudaData(void* data);
  const void* GetCudaData();

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  void* cuda_ptr_ = nullptr;  ///< CUDA data pointer.
};

}  // namespace cnstream

#endif  // CNSTREAM_SYNCMEM_CUDA_HPP_