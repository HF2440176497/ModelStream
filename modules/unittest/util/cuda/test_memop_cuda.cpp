
#include "base.hpp"

#include "cuda/cuda_check.hpp"
#include "cuda/memop_cuda.hpp"

#include "memop_factory.hpp"

namespace cnstream {


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
  EXPECT_EQ(memop->size_, bytes);
  EXPECT_EQ(allocated_mem->size_, bytes);
}

template<typename T, typename U>
std::unique_ptr<T> memop_dynamic_pointer_cast(std::unique_ptr<U>&& ptr) {
    if (!ptr) return nullptr;
    
    T* result = dynamic_cast<T*>(ptr.get());
    if (!result) return nullptr;
    
    ptr.release();
    return std::unique_ptr<T>(result);
}


/**
 * 测试与 SyncMemmory 的交互
 */
TEST(CudaMemOp, CreateSyncedMemory) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  ASSERT_NE(memop, nullptr);

  // 1. 验证 memop 是 CudaMemOp 类型
  std::unique_ptr<CudaMemOp> cuda_memop = memop_dynamic_pointer_cast<CudaMemOp>(std::move(memop));
  ASSERT_NE(cuda_memop, nullptr);
  
  size_t bytes = 64 * 4096;
  auto synced_mem = cuda_memop->CreateSyncedMemory(bytes);
  ASSERT_NE(synced_mem, nullptr);
  EXPECT_EQ(synced_mem->size_, bytes);
  EXPECT_EQ(synced_mem->GetDevId(), 0);

  // 2. 针对 sync_mem 进一步测试
  std::unique_ptr<CNSyncedMemoryCuda> cuda_synced_mem = memop_dynamic_pointer_cast<CNSyncedMemoryCuda>(std::move(synced_mem));
  ASSERT_NE(cuda_synced_mem, nullptr);
  EXPECT_EQ(cuda_synced_mem->cuda_ptr_, nullptr);
  EXPECT_EQ(cuda_synced_mem->head_, SyncedHead::UNINITIALIZED);

  EXPECT_EQ(cuda_synced_mem->own_dev_data_.size(), 2);  // CPU CUDA
  for (auto& dev_data : cuda_synced_mem->own_dev_data_) {
    EXPECT_EQ(dev_data.second, false);
  }

  auto data = cuda_memop->Allocate(bytes);
  ASSERT_NE(data, nullptr);
  ASSERT_NE(data.get(), nullptr);
  
  void* data_ptr = data.get();

  // note: 创建内存，手动填充内容，后续可以验证
  // 注意是将 host 内存填充到 cuda 内存
  ASSERT_EQ(RoundUpSize(bytes), bytes);
  void *tmp = malloc(bytes);
  const uint8_t pattern = 0xAB;
  memset(tmp, pattern, bytes);
  CHECK_CUDA_RUNTIME(cudaMemcpy(data_ptr, tmp, bytes, cudaMemcpyHostToDevice));

  // 2.1 Sync 使用外面创建的 RAII 内存
  cuda_memop->SetData(cuda_synced_mem.get(), data_ptr);
  ASSERT_EQ(cuda_synced_mem->size_, bytes);
  ASSERT_EQ(cuda_synced_mem->head_, SyncedHead::HEAD_AT_CUDA);
  EXPECT_EQ(cuda_synced_mem->own_dev_data_.size(), 2);  // CPU CUDA
  for (auto& dev_data : cuda_synced_mem->own_dev_data_) {
    EXPECT_EQ(dev_data.second, false);
  }
  ASSERT_NE(cuda_synced_mem->cuda_ptr_, nullptr);

  auto cuda_mem = dynamic_cast<CNSyncedMemoryCuda*>(cuda_synced_mem.get());
  ASSERT_NE(cuda_mem, nullptr);
  ASSERT_EQ(cuda_mem->cuda_ptr_, data_ptr);

  // 同时再测试 SyncMem 的内存同步功能
  cuda_synced_mem->ToCpu();
  ASSERT_NE(cuda_synced_mem->cpu_ptr_, nullptr);
  ASSERT_EQ(cuda_synced_mem->head_, SyncedHead::SYNCED);

  LOGI(MEMOP_TEST) << "cuda_synced_mem status: " << cuda_synced_mem->StatusToString();

  ASSERT_NE(cuda_synced_mem->GetMutableCudaData(), nullptr);
  EXPECT_EQ(cuda_synced_mem->head_, SyncedHead::HEAD_AT_CUDA);

  // 如果此时再调用 ToCpu, 不应当改变 own_dev_data_
  cuda_synced_mem->ToCpu();
  EXPECT_EQ(cuda_synced_mem->head_, SyncedHead::SYNCED);
  EXPECT_EQ(cuda_synced_mem->own_dev_data_[DevType::CPU], true);

  // 取出数据再验证
  uint8_t* cpu_data = (uint8_t*)cuda_synced_mem->cpu_ptr_;
  for (size_t i = 0; i < bytes; ++i) {
    EXPECT_EQ(cpu_data[i], pattern);
  }
  // 清理，memop 会自动清理分配的内存
  free(tmp);
}

// ------- 以下验证图像格式转换

// 指定格式，生成 uniform 图像
DecodeFrame* CreateTestDecodeFrameCuda(DataFormat fmt, int width, int height) {
  DecodeFrame* frame = new DecodeFrame(height, width, fmt);
  frame->fmt = fmt;
  frame->width = width;
  frame->height = height;
  frame->device_id = 0;
  frame->planeNum = 0;

  for (int i = 0; i < CN_MAX_PLANES; ++i) {
    frame->plane[i] = nullptr;
    frame->stride[i] = 0;
  }

  size_t frame_size = 0;
  if (fmt == DataFormat::PIXEL_FORMAT_BGR24 || 
      fmt == DataFormat::PIXEL_FORMAT_RGB24) {
    frame->planeNum = 1;
    frame_size = width * height * 3;

    uint8_t* h_data = (uint8_t*)malloc(frame_size);
    memset(h_data, 128, frame_size);

    CHECK_CUDA_RUNTIME(cudaMalloc(frame->plane[0], frame_size));
    frame->stride[0] = width * 3;
    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[0], h_data, frame_size, cudaMemcpyHostToDevice));

    free(h_data);

  } else if (fmt == DataFormat::PIXEL_FORMAT_YUV420_NV12 ||
             fmt == DataFormat::PIXEL_FORMAT_YUV420_NV21) {
              
    frame->planeNum = 2;
    size_t y_size = width * height;
    size_t uv_size = width * height / 2;
    CHECK_CUDA_RUNTIME(cudaMalloc(frame->plane[0], y_size));
    CHECK_CUDA_RUNTIME(cudaMalloc(frame->plane[1], uv_size));
    frame->stride[0] = width;
    frame->stride[1] = width;

    uint8_t* h_y = (uint8_t*)malloc(y_size);
    uint8_t* h_uv = (uint8_t*)malloc(uv_size);
    memset(h_y, 128, y_size);
    memset(h_uv, 64, uv_size);

    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[0], h_y, y_size, cudaMemcpyHostToDevice));
    CHECK_CUDA_RUNTIME(cudaMemcpy(frame->plane[1], h_uv, uv_size, cudaMemcpyHostToDevice));

    free(h_y);
    free(h_uv);
  }
  return frame;
}


void clear_decode_frame(DecodeFrame* src_frame) {
  if (src_frame->planeNum == 1) {
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[0]));
  } else if (src_frame->planeNum == 2) {
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[0]));
    CHECK_CUDA_RUNTIME(cudaFree(src_frame->plane[1]));
  } else {
    LOGF(FRAME) << "clear_decode_frame ERROR, fmt not supported.";
  }
  return;
}


TEST(CudaMemOp, ConvertImageFormat_BGR24_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::unique_ptr<CudaMemOp> cuda_memop = memop_dynamic_pointer_cast<CudaMemOp>(std::move(memop));
  ASSERT_NE(cuda_memop, nullptr);

  // 1. 填充到 frame_plane 显存）uniform data
  int width = 1280, height = 1280;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_BGR24, width, height);
  
  uint8_t* h_bgr = (uint8_t*)malloc(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    h_bgr[i * 3] = 255;     // B
    h_bgr[i * 3 + 1] = 128; // G
    h_bgr[i * 3 + 2] = 64;  // R
  }
  CHECK_CUDA_RUNTIME(cudaMemcpy(src_frame->plane[0], h_bgr, width * height * 3, cudaMemcpyHostToDevice));

  // 2. 转换到 dst_mem 显存
  size_t dst_size = width * height * 3;
  auto dst_mem = cuda_memop->Allocate(dst_size);
  void* dst_ptr = dst_mem.get();

  int ret = cuda_memop->ConvertImageFormat(dst_ptr, DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);

  // 3. 拷贝回来比较
  uint8_t* h_rgb = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb, dst_ptr, dst_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_rgb[i * 3], 64);      // R (原B)
    EXPECT_EQ(h_rgb[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(h_rgb[i * 3 + 2], 255); // B (原R)
  }

  // 4. 格式相同时，内部会拷贝
  auto dst_mem_2 = cuda_memop->Allocate(dst_size);
  void* dst_ptr_2 = dst_mem_2.get();
  ret = cuda_memop->ConvertImageFormat(dst_ptr_2, DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);
  EXPECT_NE(dst_ptr_2, dst_ptr);

  uint8_t* h_rgb_2 = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb_2, dst_ptr_2, dst_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_rgb_2[i * 3], 64);      // R (原B)
    EXPECT_EQ(h_rgb_2[i * 3 + 1], 128); // G (原G)
    EXPECT_EQ(h_rgb_2[i * 3 + 2], 255); // B (原R)
  }

  free(h_bgr);
  free(h_rgb);
  free(h_rgb_2);

  clear_decode_frame(src_frame);
  delete src_frame;
}

TEST(CudaMemOp, ConvertImageFormat_RGB24_BGR24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::unique_ptr<CudaMemOp> cuda_memop = memop_dynamic_pointer_cast<CudaMemOp>(std::move(memop));
  ASSERT_NE(cuda_memop, nullptr);

  int width = 640, height = 480;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_RGB24, width, height, cuda_memop.get());
  uint8_t* h_rgb = (uint8_t*)malloc(width * height * 3);
  for (int i = 0; i < width * height; ++i) {
    h_rgb[i * 3] = 100;     // R
    h_rgb[i * 3 + 1] = 150; // G
    h_rgb[i * 3 + 2] = 200; // B
  }
  CHECK_CUDA_RUNTIME(cudaMemcpy(src_frame->plane[0], h_rgb, width * height * 3, cudaMemcpyHostToDevice));

  size_t dst_size = width * height * 3;
  auto dst_mem = cuda_memop->Allocate(dst_size);
  void* dst_ptr = dst_mem.get();
  int ret = cuda_memop->ConvertImageFormat(dst_ptr, DataFormat::PIXEL_FORMAT_BGR24, src_frame);
  ASSERT_EQ(ret, 0);

  uint8_t* h_bgr = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_bgr, dst_ptr, dst_size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < width * height; ++i) {
    EXPECT_EQ(h_bgr[i * 3], 200);     // B (原R)
    EXPECT_EQ(h_bgr[i * 3 + 1], 150); // G (原G)
    EXPECT_EQ(h_bgr[i * 3 + 2], 100); // R (原B)
  }

  free(h_rgb);
  free(h_bgr);

  clear_decode_frame(src_frame);
  delete src_frame;
}


TEST(CudaMemOp, ConvertImageFormat_NV12_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::unique_ptr<CudaMemOp> cuda_memop = memop_dynamic_pointer_cast<CudaMemOp>(std::move(memop));
  ASSERT_NE(cuda_memop, nullptr);

  int width = 1920, height = 1080;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_YUV420_NV12, width, height, cuda_memop.get());

  size_t dst_size = width * height * 3;
  auto dst_mem = cuda_memop->Allocate(dst_size);
  void* dst_ptr = dst_mem.get();
  int ret = cuda_memop->ConvertImageFormat(dst_ptr, DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);

  uint8_t* h_rgb = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb, dst_ptr, dst_size, cudaMemcpyDeviceToHost));
  EXPECT_NE(h_rgb, nullptr);
  free(h_rgb);

  clear_decode_frame(src_frame);
  delete src_frame;
}

TEST(CudaMemOp, ConvertImageFormat_NV21_RGB24) {
  auto& factory = MemOpFactory::Instance();
  auto memop = factory.CreateMemOp(DevType::CUDA, 0);
  std::unique_ptr<CudaMemOp> cuda_memop = memop_dynamic_pointer_cast<CudaMemOp>(std::move(memop));
  ASSERT_NE(cuda_memop, nullptr);
  
  int width = 1920, height = 1080;
  DecodeFrame* src_frame = CreateTestDecodeFrameCuda(DataFormat::PIXEL_FORMAT_YUV420_NV21, width, height, cuda_memop.get());

  size_t dst_size = width * height * 3;
  auto dst_mem = cuda_memop->Allocate(dst_size);
  void* dst_ptr = dst_mem.get();
  int ret = cuda_memop->ConvertImageFormat(dst_ptr, DataFormat::PIXEL_FORMAT_RGB24, src_frame);
  ASSERT_EQ(ret, 0);

  uint8_t* h_rgb = (uint8_t*)malloc(dst_size);
  CHECK_CUDA_RUNTIME(cudaMemcpy(h_rgb, dst_ptr, dst_size, cudaMemcpyDeviceToHost));
  EXPECT_NE(h_rgb, nullptr);
  free(h_rgb);

  clear_decode_frame(src_frame);
  delete src_frame;
}

}  // end namespace cnstream