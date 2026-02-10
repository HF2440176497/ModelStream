// cuda_memop.hpp
#ifndef MEMOP_CUDA_HPP_
#define MEMOP_CUDA_HPP_

#include "memop.hpp"

#include "cnstream_allocator.hpp"
#include "cnstream_sysncmem.hpp"
#include "cuda/cnstream_allocator_cuda.hpp"
#include "cuda/cnstream_sysncmem_cuda.hpp"


namespace cnstream {

class CudaMemOp : public MemOp {
 public:
  CudaMemOp();
  ~CudaMemOp() override;
  
  explicit CudaMemOp(int device_id) : device_id_(device_id) {}
  int GetDeviceId() const override { return device_id_; }
  std::shared_ptr<CNSyncedMemory> CreateSyncedMemory(size_t size) override;
  std::shared_ptr<void> Allocate(size_t bytes) override;
  void Copy(void* dst, const void* src, size_t size) override;
  void SetData(std::shared_ptr<CNSyncedMemory> mem, void* data) override;
  int ConvertImageFormat(void* dst, DataFormat dst_fmt, const DecodeFrame* src_frame) override;

 protected:
  int device_id_;
};

}  // namespace cnstream

#endif  // CUDA_MEMOP_HPP_