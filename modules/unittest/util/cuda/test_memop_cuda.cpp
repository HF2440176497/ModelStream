
#include "base.hpp"


#include "util/cuda/cuda_check.hpp"  // framework

#include "util/include/cuda/memop_cuda.hpp"
#include "util/include/memop_factory.hpp"

namespace cnstream {

// 单卡设备的 Memop 创建
TEST(CudaMemOpFactory, RegisterCudaMemOpCreator) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_TRUE(memop != nullptr);
  EXPECT_EQ(memop->GetDeviceId(), 0);
}


TEST(CudaMemOp, Allocate) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_NE(memop, nullptr);

  size_t bytes = 64 * 4096;
  auto allocated_mem = memop->Allocate(bytes);
  ASSERT_NE(allocated_mem, nullptr);
  ASSERT_NE(allocated_mem.get(), nullptr);
}

/**
 * 测试与 SyncMemmory 的交互
 */
TEST(CudaMemOp, CreateSyncedMemory) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_NE(memop, nullptr);
  
  size_t bytes = 64 * 4096;
  auto synced_mem = memop->CreateSyncedMemory(bytes);
  ASSERT_NE(synced_mem, nullptr);

  auto data = memop->Allocate(bytes);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(data.get(), nullptr);
  
  void* data_ptr = data.get();

  // note: 创建内存，手动填充内容，后续可以验证
  ASSERT_EQ(RoundUpSize(bytes), bytes);
  void *tmp = malloc(bytes);
  const uint8_t pattern = 0xAB;
  memset(tmp, pattern, bytes);
  CHECK_CUDA_RUNTIME(cudaMemcpy(data_ptr, tmp, bytes, cudaMemcpyHostToDevice));

  memop->SetData(synced_mem, data_ptr);
  ASSERT_EQ(synced_mem->size_, bytes);
  ASSERT_EQ(synced_mem->head_, SyncedHead::HEAD_AT_CUDA);

  auto cuda_mem = std::dynamic_pointer_cast<CNSyncedMemoryCuda>(synced_mem);
  ASSERT_NE(cuda_mem, nullptr);
  ASSERT_EQ(cuda_mem->cuda_ptr_, data_ptr);

  // 同时再测试 SyncMem 的内存同步功能
  synced_mem->ToCpu();
  ASSERT_NE(synced_mem->cpu_ptr_, nullptr);
  ASSERT_EQ(synced_mem->head_, SyncedHead::SYNCED);

  // 取出数据再验证
  uint8_t* cpu_data = (uint8_t*)synced_mem->cpu_ptr_;
  for (size_t i = 0; i < bytes; ++i) {
    EXPECT_EQ(cpu_data[i], pattern);
  }
  // 清理，memop 会自动清理分配的内存
  free(tmp);
}

// CUDA 暂不测试图像格式转换

}  // end namespace cnstream