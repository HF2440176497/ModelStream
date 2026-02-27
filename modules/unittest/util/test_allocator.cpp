
#include "base.hpp"
#include "cnstream_allocator.hpp"

namespace cnstream {

TEST(Allocator, CpuMemAlloc) {
  auto allocator = std::make_shared<CpuAllocator>();

  std::shared_ptr<void> mem = cnMemAlloc(4000, allocator);  // align to 4096 byte
  ASSERT_NE(mem, nullptr);
  void* ptr = mem.get();
  ASSERT_NE(ptr, nullptr);

  EXPECT_EQ(allocator->device_id_, -1);
  EXPECT_EQ(allocator->size_, 4096);

  auto mem2 = cnMemAlloc(4097, allocator);
  ASSERT_NE(mem2, nullptr);
  void* ptr2 = mem2.get();
  ASSERT_NE(ptr2, nullptr);
  EXPECT_EQ(allocator->size_, 4096 * 2);

}

}