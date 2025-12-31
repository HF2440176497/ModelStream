
#include "util/cnstream_allocator.hpp"
#include "include/base.hpp"  // unittest

#ifdef NVIDIA_PLATFORM
#include "util/cuda/cnstream_allocator_cuda.hpp"  // framework
#endif


class TestCpuAllocator : public MemoryAllocator {
 public:
  TestCpuAllocator() : MemoryAllocator(-1) {}
  ~TestCpuAllocator() = default;

  void *alloc(size_t size, int timeout_ms = 0) override {
    size_t alloc_size = (size + 4095) & (~0xFFF);  // Align 4096
    last_size_ = alloc_size;
    return static_cast<void *>(new (std::nothrow) unsigned char[alloc_size]);
  }
  void free(void *p) override {
    unsigned char *ptr = static_cast<unsigned char *>(p);
    delete[] ptr;
  }
 public:
  size_t last_size_;
};


TEST(Allocator, CpuMemAlloc) {
  auto allocator = std::make_shared<TestCpuAllocator>();
  std::shared_ptr<void> mem = cnMemAlloc(4000, allocator);
  ASSERT_TRUE(mem != nullptr);
  void* ptr = mem.get();
  ASSERT_TRUE(ptr != nullptr);
  ASSERT_EQ(allocator->last_size_, 4096);

  auto mem2 = cnMemAlloc(4097, allocator);
  ASSERT_TRUE(mem2 != nullptr);
  void* ptr2 = mem2.get();
  ASSERT_TRUE(ptr2 != nullptr);
  ASSERT_EQ(allocator->last_size_, int(4096 * 2));  // 应当对齐到 4096 bytes
}

// 英伟达平台内存分配器测试
#ifdef NVIDIA_PLATFORM



#endif