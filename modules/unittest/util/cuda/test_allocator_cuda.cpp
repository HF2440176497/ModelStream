
#include "base.hpp"
#include "cnstream_allocator.hpp"  // modules/util/include

#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_allocator_cuda.hpp"

namespace cnstream {

TEST(Allocator, CudaMemAlloc) {
  auto allocator = std::make_shared<CudaAllocator>();
  std::shared_ptr<void> mem = cnMemAlloc(4000, allocator);  // align to 4096 bytes
  ASSERT_NE(mem, nullptr);
  void* ptr = mem.get();
  ASSERT_NE(ptr, nullptr);
  auto mem2 = cnMemAlloc(4097, allocator);
  ASSERT_NE(mem2, nullptr);
  void* ptr2 = mem2.get();
  ASSERT_NE(ptr2, nullptr);
}

}
