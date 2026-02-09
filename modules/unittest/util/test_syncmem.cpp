/**
 * @file test_syncmem.cpp
 *
 * This file contains gtest unit tests for the CNSyncedMemory class on CUDA platform.
 */

#include "base.hpp"
#include <vector>
#include <cstring>


#include "cnstream_syncmem_cuda.hpp"
#include "cnstream_logging.hpp"



class CNSyncedMemoryTest : public ::testing::Test {
protected:
  static const int kTestSize = 1024;
  static const int kFloatCount = kTestSize / sizeof(float);
};

namespace cnstream {

/**
 * 在 CPU 上设置数据 —— 转移到 CUDA —— 转移回 CPU 并验证
 */
TEST_F(CNSyncedMemoryTest, BasicFunctionality) {
  CNSyncedMemoryCuda mem(kTestSize);

  // Set data on CPU
  float* cpu_data = static_cast<float*>(mem.GetMutableCpuData());
  ASSERT_NE(cpu_data, nullptr) << "Failed to get mutable CPU data";
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CPU) << "Head should be HEAD_AT_CPU";
  for (int i = 0; i < kFloatCount; i++) {
    cpu_data[i] = static_cast<float>(i);
  }
  // Transfer to CUDA
  mem.ToCuda();
  ASSERT_EQ(mem.GetHead(), SyncedHead::SYNCED) << "Head should be SYNCED";
  ASSERT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data should be owned";
  ASSERT_TRUE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data should be owned";

  // Verify data on CUDA
  const float* cuda_data = static_cast<const float*>(mem.GetCudaData());
  ASSERT_NE(cuda_data, nullptr) << "Failed to get CUDA data";

  // Transfer back to CPU and verify
  float* cpu_data2 = static_cast<float*>(mem.GetMutableCpuData());
  ASSERT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CPU) << "Head should be HEAD_AT_CPU after GetMutableCpuData";
  ASSERT_NE(cpu_data2, nullptr) << "Failed to get mutable CPU data after CUDA transfer";

  for (int i = 0; i < kFloatCount; i++) {
    ASSERT_FLOAT_EQ(cpu_data2[i], static_cast<float>(i)) << "Data mismatch at index " << i;
  }
}

// Test 2: Device context
TEST_F(CNSyncedMemoryTest, DeviceContext) {
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    GTEST_SKIP() << "No CUDA devices found, skipping device context test";
  }
  const int dev_id = 0; // Use first CUDA device
  CNSyncedMemoryCuda mem(kTestSize, dev_id);
  EXPECT_EQ(mem.GetDevId(), dev_id) << "Device ID mismatch";

  float* cpu_data = static_cast<float*>(mem.GetMutableCpuData());
  ASSERT_NE(cpu_data, nullptr) << "Failed to get mutable CPU data";

  for (int i = 0; i < kFloatCount; i++) {
    cpu_data[i] = static_cast<float>(i * 2);
  }
  mem.ToCuda();
  const float* cuda_data = static_cast<const float*>(mem.GetCudaData());
  ASSERT_NE(cuda_data, nullptr) << "Data transfer with device context failed";
}

TEST_F(CNSyncedMemoryTest, MemoryManagement) {
  CNSyncedMemoryCuda mem(kTestSize);

  float* manual_cuda_ptr;
  cudaError_t result = cudaMalloc(&manual_cuda_ptr, kTestSize);
  ASSERT_EQ(result, cudaSuccess) << "Failed to allocate CUDA memory manually";

  mem.SetCudaData(manual_cuda_ptr);
  EXPECT_FALSE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data ownership should be false after SetCudaData";
  EXPECT_EQ(mem.GetHead(), SyncedHead::HEAD_AT_CUDA) << "Head should be HEAD_AT_CUDA after SetCudaData";

  const float* cuda_data = static_cast<const float*>(mem.GetCudaData());
  EXPECT_EQ(cuda_data, manual_cuda_ptr) << "Manual CUDA pointer not set correctly";

  EXPECT_EQ(mem.cpu_ptr_, nullptr);
  mem.ToCpu();
  EXPECT_NE(mem.cpu_ptr_, nullptr) << "CPU data pointer should not be NULL after ToCpu";
  EXPECT_TRUE(mem.own_dev_data_[DevType::CPU]) << "CPU data ownership should be true after ToCpu";
  EXPECT_FALSE(mem.own_dev_data_[DevType::CUDA]) << "CUDA data ownership should be true after ToCpu";
  EXPECT_EQ(mem.GetHead(), SyncedHead::SYNCED) << "Head should be HEAD_AT_CPU after ToCpu";

  cudaFree(manual_cuda_ptr);
}

}