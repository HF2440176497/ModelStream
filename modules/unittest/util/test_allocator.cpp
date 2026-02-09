
#include "include/base.hpp"
#include "util/cnstream_allocator.hpp"

namespace cnstream {

TEST(Allocator, CpuMemAlloc) {
  auto allocator = std::make_shared<CpuAllocator>();
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