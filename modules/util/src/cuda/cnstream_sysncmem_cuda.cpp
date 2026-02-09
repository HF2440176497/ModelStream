
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_sysncmem_cuda.hpp"

namespace cnstream {

CNSyncedMemoryCuda::CNSyncedMemoryCuda(size_t size) : CNSyncedMemory(size) {
  own_dev_data_[DevType::CPU] = false;
  own_dev_data_[DevType::CUDA] = false;
}

CNSyncedMemoryCuda::CNSyncedMemoryCuda(size_t size, int dev_id) : CNSyncedMemory(size) {
  std::lock_guard<std::mutex> lock(mutex_);
  int device_count = 0;
  CHECK_CUDA_RUNTIME(cudaGetDeviceCount(&device_count));
  if (dev_id < 0 || dev_id >= device_count) {
    LOGF(FRAME) << "Invalid CUDA device id: " << dev_id << ", available devices: " << device_count;
    dev_id = 0;
  }
  dev_id_ = dev_id;
  own_dev_data_[DevType::CPU] = false;
  own_dev_data_[DevType::CUDA] = false;
}

CNSyncedMemoryCuda::~CNSyncedMemoryCuda() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (0 == size_) return;
  if (cpu_ptr_ && own_dev_data_[DevType::CPU]) {
    free(cpu_ptr_);
  }
  if (cuda_ptr_ && own_dev_data_[DevType::CUDA]) {
    // Set device before freeing memory
    if (dev_id_ >= 0) {
      CHECK_CUDA_RUNTIME(cudaSetDevice(dev_id_));
    }
    CHECK_CUDA_RUNTIME(cudaFree(cuda_ptr_));
  }
}

void CNSyncedMemoryCuda::ToCpu() {
  if (0 == size_) return;
  
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      if (cuda_ptr_ or cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cuda_ptr_ and cpu_ptr_ should be NULL.";
        return;
      }
      CHECK_CUDA_RUNTIME(cudaMallocHost(&cpu_ptr_, size_));
      memset(cpu_ptr_, 0, size_);
      head_ = SyncedHead::HEAD_AT_CPU;
      own_dev_data_[DevType::CPU] = true;
      break;
    case SyncedHead::HEAD_AT_CUDA:
      if (NULL == cuda_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCpu ERROR, cuda_ptr_ should not be NULL.";
        return;
      }
      if (NULL == cpu_ptr_) {
        CHECK_CUDA_RUNTIME(cudaMallocHost(&cpu_ptr_, size_));
        memset(cpu_ptr_, 0, size_);
        own_dev_data_[DevType::CPU] = true;
      }
      // Sasha: 如果 cpu_ptr_ 是指定的，上面判断不通过，那么我们不应该改变 own_dev_data_ 对应标记
      // Set device if specified
      if (dev_id_ >= 0) {
        cudaSetDevice(dev_id_);
      }
      CHECK_CUDA_RUNTIME(cudaMemcpy(cpu_ptr_, cuda_ptr_, size_, cudaMemcpyDeviceToHost));
      head_ = SyncedHead::SYNCED;
      break;
    case SyncedHead::HEAD_AT_CPU:
    case SyncedHead::SYNCED:
      break;
  }
}

void CNSyncedMemoryCuda::ToCuda() {
  if (0 == size_) return;
  // Set device if specified
  if (dev_id_ >= 0) {
    cudaSetDevice(dev_id_);
  }
  switch (head_) {
    case SyncedHead::UNINITIALIZED:
      if (cuda_ptr_ or cpu_ptr_) {  // 不应该存在已分配的 CUDA 内存
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCuda ERROR, cuda_ptr_ and cpu_ptr_ should be NULL.";
        return;
      }
      CHECK_CUDA_RUNTIME(cudaMalloc(&cuda_ptr_, size_));
      head_ = SyncedHead::HEAD_AT_CUDA;
      own_dev_data_[DevType::CUDA] = true;
      break;
    case SyncedHead::HEAD_AT_CPU:
      if (NULL == cpu_ptr_) {
        LOGE(FRAME) << "CNSyncedMemoryCuda::ToCuda ERROR, cpu_ptr_ should not be NULL.";
        return;
      }
      if (NULL == cuda_ptr_) {
        CHECK_CUDA_RUNTIME(cudaMalloc(&cuda_ptr_, size_));
        own_dev_data_[DevType::CUDA] = true;
      }
      CHECK_CUDA_RUNTIME(cudaMemcpy(cuda_ptr_, cpu_ptr_, size_, cudaMemcpyHostToDevice));
      head_ = SyncedHead::SYNCED;
      break;
    case SyncedHead::HEAD_AT_CUDA:
    case SyncedHead::SYNCED:
      break;
  }
}

const void* CNSyncedMemoryCuda::GetCudaData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCuda();
  return const_cast<const void*>(cuda_ptr_);
}

void CNSyncedMemoryCuda::SetCudaData(void* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (0 == size_) return;
  LOGF_IF(FRAME, nullptr == data) << "data is NULL.";
  if (own_dev_data_[DevType::CUDA]) {
    if (!cuda_ptr_) {
      LOGE(FRAME) << "CNSyncedMemoryCuda::SetCudaData ERROR, cuda_ptr_ should not be NULL.";
      return;
    }
    if (dev_id_ >= 0) {
      CHECK_CUDA_RUNTIME(cudaSetDevice(dev_id_));
    }
    CHECK_CUDA_RUNTIME(cudaFree(cuda_ptr_));
  }
  cuda_ptr_ = data;
  head_ = SyncedHead::HEAD_AT_CUDA;
  own_dev_data_[DevType::CUDA] = false;
}

void* CNSyncedMemory::GetMutableCudaData() {
  std::lock_guard<std::mutex> lock(mutex_);
  ToCuda();
  head_ = SyncedHead::HEAD_AT_CUDA;
  return cuda_ptr_;
}

}