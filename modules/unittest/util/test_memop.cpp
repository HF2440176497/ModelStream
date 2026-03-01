
#include "data_source_param.hpp"
#include "memop_factory.hpp"
#include "memop.hpp"

#include "base.hpp"

namespace cnstream {

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
  frame->planeNum = 0;

  for (int i = 0; i < CN_MAX_PLANES; ++i) {
    frame->plane[i] = nullptr;
    frame->stride[i] = 0;
  }

  const uint8_t R = 10, G = 128, B = 242;
  const uint8_t Y = 111;
  const uint8_t U = 190;
  const uint8_t V = 72;

  size_t frame_size = 0;
  if (fmt == DataFormat::PIXEL_FORMAT_BGR24 || 
      fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    frame->planeNum = 1;
    frame_size = width * height * 3;
    frame->plane[0] = malloc(frame_size);
    uint8_t* data = static_cast<uint8_t*>(frame->plane[0]);
    for (int i = 0; i < width * height; ++i) {
      if (fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        data[i * 3] = B;
        data[i * 3 + 1] = G;
        data[i * 3 + 2] = R;
      } else {
        data[i * 3] = R;
        data[i * 3 + 1] = G;
        data[i * 3 + 2] = B;
      }
    }
  } else if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12 ||
             fmt == DataFormat::PIXEL_FORMAT_YUV420_NV21) {
    frame->planeNum = 2;
    frame_size = width * height * 3 / 2;
    frame->plane[0] = malloc(width * height);
    frame->plane[1] = malloc(width * height / 2);
    memset(frame->plane[0], Y, width * height);
    uint8_t* uv_data = static_cast<uint8_t*>(frame->plane[1]);
    for (size_t i = 0; i < static_cast<size_t>(width * height / 2); i += 2) {
      if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12) {
        uv_data[i] = U;
        uv_data[i + 1] = V;
      } else {
        uv_data[i] = V;
        uv_data[i + 1] = U;
      }
    }
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


TEST(MemOpFactory, RegisterMemOpCreator) {
  auto& factory = MemOpFactory::Instance();
  factory.PrintRegisteredCreators();  // 此时会显示 CPU MemOp
  auto memop = factory.CreateMemOp(DevType::CPU, -1);  // CreateMemOp 查找已注册的 CPU memop
  ASSERT_TRUE(memop != nullptr);
}


TEST(MemOp, CreateMemOp) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CPU, -1);
  ASSERT_NE(memop, nullptr);
  size_t bytes = 64 * 1024;

  auto synced_mem = memop->CreateSyncedMemory(bytes);
  ASSERT_NE(synced_mem, nullptr);
  EXPECT_EQ(synced_mem->GetSize(), bytes);
  EXPECT_EQ(synced_mem->GetDevId(), -1);
  EXPECT_FALSE(synced_mem->own_dev_data_[DevType::CPU]);

  void* data = synced_mem->Allocate();
  ASSERT_NE(data, nullptr);
  ASSERT_TRUE(synced_mem->own_dev_data_[DevType::CPU]);
  ASSERT_EQ(synced_mem->GetHead(), SyncedHead::HEAD_AT_CPU);
  ASSERT_EQ(synced_mem->GetMutableCpuData(), data);
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
  auto dst_mem = memop->CreateSyncedMemory(dst_size);
  ASSERT_NE(dst_mem, nullptr);
  
  int ret = memop->ConvertImageFormat(dst_mem.get(), DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);
  
  void* rgb_data = const_cast<void*>(dst_mem->GetCpuData());
  uint8_t* rgb_data_8 = static_cast<uint8_t*>(rgb_data);
  
  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(rgb_data_8[i * 3], 64);     // R (原B)
    EXPECT_EQ(rgb_data_8[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(rgb_data_8[i * 3 + 2], 255); // B (原R)
  }
  CleanupTestDecodeFrame(src_frame);
}

}