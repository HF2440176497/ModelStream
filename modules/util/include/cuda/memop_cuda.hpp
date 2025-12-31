// cuda_memop.hpp
#ifndef MEMOP_CUDA_HPP_
#define MEMOP_CUDA_HPP_

#include "cnstream_logging.hpp"  // framework

#include "util/include/memop.hpp"
#include "util/include/memop_factory.hpp"
#include "util/include/cnstream_sysncmem.hpp"
#include "util/include/cuda/cnstream_allocator_cuda.hpp"


namespace cnstream {

class CudaMemOp : public MemOp {
 public:
  explicit CudaMemOp(int device_id) : device_id_(device_id) {}
  int GetDeviceId() const override { return device_id_; }
  std::shared_ptr<CNSyncedMemory> CreateSyncedMemory(size_t size) override;
  std::shared_ptr<void> Allocate(size_t bytes) override;
  void Copy(void* dst, const void* src, size_t size) override;
  void SetData(std::shared_ptr<CNSyncedMemory> mem, void* data) override;
  int ConvertImageFormat(void* dst, CNDataFormat dst_fmt, const DecodeFrame* src_frame) override;

 protected:
  int device_id_;
};

static bool RegisterCudaMemOp() {
  auto& factory = MemOpFactory::Instance();
  bool result = true;
  result &= factory.RegisterMemOpCreator(DevType::CUDA,
    [](int dev_id) {
      return std::make_unique<CudaMemOp>(dev_id);
    });
  return result;
}

static bool cuda_memops_registered = RegisterCudaMemOp();

}  // namespace cnstream

#endif  // CUDA_MEMOP_HPP_