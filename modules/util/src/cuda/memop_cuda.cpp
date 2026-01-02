// cuda_memop.cpp

#include "cuda/memop_cuda.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_sysncmem_cuda.hpp"

namespace cnstream {

std::shared_ptr<CNSyncedMemory> CudaMemOp::CreateSyncedMemory(size_t size) {
  return std::make_shared<CNSyncedMemoryCuda>(size, device_id_);
}

std::shared_ptr<void> CudaMemOp::Allocate(size_t bytes) {
  bytes = RoundUpSize(bytes);
  CudaDeviceGuard guard(device_id_);
  return cnCudaMemAlloc(bytes, device_id_);
}

void CudaMemOp::Copy(void* dst, const void* src, size_t size) {
  CudaDeviceGuard guard(device_id_);
  CHECK_CUDA_RUNTIME(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void CudaMemOp::SetData(std::shared_ptr<CNSyncedMemory> mem, void* data) {
  auto cuda_mem = std::dynamic_pointer_cast<CNSyncedMemoryCuda>(mem);
  if (!cuda_mem) {
    throw std::runtime_error("CudaMemOp: mem is not CNSyncedMemoryCuda");
  }
  CudaDeviceGuard guard(device_id_);
  cuda_mem->SetCudaData(data);
}

/**
 * @brief 使用 CUDA, 将解码帧转换为 dst_fmt 格式
 */
int CudaMemOp::ConvertImageFormat(void* dst, CNDataFormat dst_fmt, const DecodeFrame* src_frame) {


  return 0;
}


}  // namespace cnstream