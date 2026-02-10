

#include "util/include/memop_factory.hpp"
#include "util/include/memop.hpp"

#include "base.hpp"

/**
 * @brief 创建一个测试用的 DecodeFrame
 * @param fmt 图像格式
 * @param width 图像宽度
 * @param height 图像高度
 * @return 返回一个指向测试用 DecodeFrame 的指针
 */
DecodeFrame* CreateTestDecodeFrame(DataFormat fmt, int width, int height) {
  DecodeFrame* frame = new DecodeFrame(height, width, fmt);
  frame->fmt = fmt;
  frame->width = width;
  frame->height = height;
  frame->device_id = -1;
  
  size_t frame_size = 0;
  if (fmt == DataFormat::PIXEL_FORMAT_BGR24 || 
      fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    frame->planeNum = 1;
    frame_size = width * height * 3;
    frame->plane[0] = malloc(frame_size);
    memset(frame->plane[0], 128, frame_size);
  } else if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12 ||
             fmt == DataFormat::PIXEL_FORMAT_YUV420_NV21) {
    frame->planeNum = 2;
    frame_size = width * height * 3 / 2;
    frame->plane[0] = malloc(width * height);  // Y plane
    frame->plane[1] = malloc(width * height / 2);  // UV plane
    memset(frame->plane[0], 128, width * height);
    memset(frame->plane[1], 64, width * height / 2);
  }
  return frame;
}

// 辅助函数：清理测试用的DecodeFrame
void CleanupTestDecodeFrame(DecodeFrame* frame) {
  if (frame) {
    if (frame->plane[0]) free(frame->plane[0]);
    if (frame->plane[1]) free(frame->plane[1]);
    delete frame;
  }
}

namespace cnstream {

TEST(MemOpFactory, RegisterMemOpCreator) {
  auto& factory = MemOpFactory::Instance();
  factory.PrintRegisteredCreators();  // 此时会显示 CPU MemOp
  auto memop = factory.CreateMemOp(DevType::CPU, -1);  // CreateMemOp 查找已注册的 CPU memop
  ASSERT_TRUE(memop != nullptr);
}


TEST(MemOp, CreateMemOp) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CPU, -1);
  ASSERT_TRUE(memop != nullptr);

  // Mem Manager
  auto mem_manager = cnstream::MemoryBufferCollection();
  Buffer& buffer = mem_manager.GetBuffer(DevType::CPU, bytes);
  EXPECT_TRUE(mem_manager.Has(DevType::CPU));
  ASSERT_EQ(buffer.data, nullptr);  // 首先注册时，Data 为空

  size_t bytes = 64 * 1024;
  auto synced_mem = memop->CreateSyncedMemory(bytes);
  ASSERT_TRUE(synced_mem != nullptr);

  auto data = memop->Allocate(bytes);  // std::shared_ptr<void>
  ASSERT_TRUE(data != nullptr);
  void* data_void_ptr = data.get();
  ASSERT_TRUE(data_void_ptr != nullptr);

  memop->SetData(synced_mem, data_void_ptr);
  buffer.data = std::move(data);
  data = nullptr;
  ASSERT_NE(buffer.data, nullptr);

  // 2. 尝试再申请一个新 Buffer
  size_t test_size = 2 * bytes;  
  Buffer& new_buffer = mem_manager.GetBuffer(DevType::CPU, test_size, -1);
  EXPECT_EQ(new_buffer.size, test_size);
  EXPECT_EQ(new_buffer.device_id, -1);
  EXPECT_TRUE(mem_manager.Has(DevType::CPU));
  EXPECT_EQ(mem_manager.GetDeviceCount(), 1);  // 只会含有一个 CPU Buffer
  EXPECT_NE(&new_buffer, &buffer);  // 两者地址不同
  
  // 重新取出并对比 应该等于第二次放入的更大的 Buffer
  Buffer* get_buffer = mem_manager.Get(DevType::CPU);
  ASSERT_TRUE(get_buffer != nullptr);
  EXPECT_EQ(get_buffer->size, test_size);
  
  // 测试清理
  mem_manager.Clear(DevType::CPU);
  EXPECT_FALSE(mem_manager.Has(DevType::CPU));
  EXPECT_EQ(mem_manager.GetDeviceCount(), 0);
  
  mem_manager.ClearAll();
}

/**
 * @brief 测试图像格式转换
 */
TEST(MemOp, ConvertImageFormat_BGR24_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CPU, -1);
  ASSERT_TRUE(memop != nullptr);
  
  int width = 1280, height = 1280;
  DecodeFrame* src_frame = CreateTestDecodeFrame(DataFormat::PIXEL_FORMAT_BGR24, width, height);
  uint8_t* bgr_data = static_cast<uint8_t*>(src_frame->plane[0]);
  for (int i = 0; i < width * height; ++i) {
    bgr_data[i * 3] = 255;     // B
    bgr_data[i * 3 + 1] = 128; // G
    bgr_data[i * 3 + 2] = 64;  // R
  }
  
  size_t dst_size = width * height * 3;
  auto dst_mem = memop->Allocate(dst_size);
  void* dst_ptr = dst_mem.get();
  int ret = memop->ConvertImageFormat(dst_ptr, DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  EXPECT_EQ(ret, 0);
  
  // 验证转换结果
  uint8_t* rgb_data = static_cast<uint8_t*>(dst_ptr);
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(rgb_data[i * 3], 64);     // R (原B)
    EXPECT_EQ(rgb_data[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(rgb_data[i * 3 + 2], 255); // B (原R)
  }
  CleanupTestDecodeFrame(src_frame);
}

}