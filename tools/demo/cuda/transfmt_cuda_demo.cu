/**
 * @file cuda_transfmt.cpp
 * @brief CUDA 图像格式转换 Demo
 *
 * 本 Demo 演示如何使用 CUDA 在 GPU 显存上进行图像格式转换
 * 支持 NV12/YUV420 -> RGB24/BGR24 的转换
 */

#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <vector>

#include "libyuv/convert.h"
#include "libyuv/convert_from_argb.h"


#define DEFAULT_IMAGE_PATH "test_image.png"

#define CHECK_CUDA_RUNTIME(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
  if (code != cudaSuccess) {
    const char* err_name = cudaGetErrorName(code);
    const char* err_message = cudaGetErrorString(code);
    printf("check_cuda_runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", 
		file, line, op, err_name, err_message);
    return false;
  }
  return true;
}

#define CHECK_CUDA_KERNEL(...)                                   \
  __VA_ARGS__;                                                   \
  do {                                                           \
    cudaError_t cudaStatus = cudaPeekAtLastError();              \
    if (cudaStatus != cudaSuccess) {                             \
      INFO("launch failed: %s", cudaGetErrorString(cudaStatus)); \
    }                                                            \
  } while (0);

enum class DataFormat {
  INVALID = -1,                 /*!< This frame is invalid. */
  PIXEL_FORMAT_YUV420_NV21 = 0, /*!< This frame is in the YUV420SP(NV21) format. */
  PIXEL_FORMAT_YUV420_NV12,     /*!< This frame is in the YUV420sp(NV12) format. */
  PIXEL_FORMAT_BGR24,           /*!< This frame is in the BGR24 format. */
  PIXEL_FORMAT_RGB24,           /*!< This frame is in the RGB24 format. */
  PIXEL_FORMAT_ARGB32,          /*!< This frame is in the ARGB32 format. */
  PIXEL_FORMAT_ABGR32,          /*!< This frame is in the ABGR32 format. */
  PIXEL_FORMAT_RGBA32,          /*!< This frame is in the RGBA32 format. */
  PIXEL_FORMAT_BGRA32           /*!< This frame is in the BGRA32 format. */
};

__global__ void NV12ToRGB24Kernel(const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane,
                                  uint8_t* __restrict__ rgb_out, int width, int height, int y_stride, int uv_stride,
                                  int rgb_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index
  int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index

  if (x >= width || y >= height) return;

  int y_idx = y * y_stride + x;
  int uv_idx = (y / 2) * uv_stride + (x / 2) * 2;

  int y_val = y_plane[y_idx];
  int u_val = uv_plane[uv_idx + 1] - 128;
  int v_val = uv_plane[uv_idx] - 128;

  int r = y_val + (int)(1.402f * v_val);
  int g = y_val - (int)(0.344f * u_val + 0.714f * v_val);
  int b = y_val + (int)(1.772f * u_val);

  r = r < 0 ? 0 : (r > 255 ? 255 : r);
  g = g < 0 ? 0 : (g > 255 ? 255 : g);
  b = b < 0 ? 0 : (b > 255 ? 255 : b);

  int rgb_idx = y * rgb_stride + x * 3;
  rgb_out[rgb_idx + 0] = static_cast<uint8_t>(r);
  rgb_out[rgb_idx + 1] = static_cast<uint8_t>(g);
  rgb_out[rgb_idx + 2] = static_cast<uint8_t>(b);
}

__global__ void NV12ToBGR24Kernel(const uint8_t* __restrict__ y_plane, const uint8_t* __restrict__ uv_plane,
                                  uint8_t* __restrict__ bgr_out, int width, int height, int y_stride, int uv_stride,
                                  int bgr_stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int y_idx = y * y_stride + x;
  int uv_idx = (y / 2) * uv_stride + (x / 2) * 2;

  int y_val = y_plane[y_idx];
  int u_val = uv_plane[uv_idx + 1] - 128;
  int v_val = uv_plane[uv_idx] - 128;

  int r = y_val + (int)(1.402f * v_val);
  int g = y_val - (int)(0.344f * u_val + 0.714f * v_val);
  int b = y_val + (int)(1.772f * u_val);

  r = r < 0 ? 0 : (r > 255 ? 255 : r);
  g = g < 0 ? 0 : (g > 255 ? 255 : g);
  b = b < 0 ? 0 : (b > 255 ? 255 : b);

  int bgr_idx = y * bgr_stride + x * 3;
  bgr_out[bgr_idx + 0] = static_cast<uint8_t>(b);
  bgr_out[bgr_idx + 1] = static_cast<uint8_t>(g);
  bgr_out[bgr_idx + 2] = static_cast<uint8_t>(r);
}

__global__ void RGB24ToBGR24Kernel(const uint8_t* __restrict__ rgb_in, uint8_t* __restrict__ bgr_out, int width,
                                   int height, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * stride + x * 3;
  bgr_out[idx + 0] = rgb_in[idx + 2];
  bgr_out[idx + 1] = rgb_in[idx + 1];
  bgr_out[idx + 2] = rgb_in[idx + 0];
}

__global__ void BGR24ToRGB24Kernel(const uint8_t* __restrict__ bgr_in, uint8_t* __restrict__ rgb_out, int width,
                                   int height, int stride) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;

  int idx = y * stride + x * 3;
  rgb_out[idx + 0] = bgr_in[idx + 2];
  rgb_out[idx + 1] = bgr_in[idx + 1];
  rgb_out[idx + 2] = bgr_in[idx + 0];
}

struct TestFrame {
  int                  width;
  int                  height;
  DataFormat           fmt;
  std::vector<uint8_t> y_plane;
  std::vector<uint8_t> uv_plane;
  std::vector<uint8_t> rgb_plane;
  std::vector<uint8_t> bgr_plane;

  void* d_y_plane = nullptr;
  void* d_uv_plane = nullptr;
  void* d_rgb_plane = nullptr;
  void* d_bgr_plane = nullptr;

  ~TestFrame() {
    if (d_y_plane) cudaFree(d_y_plane);
    if (d_uv_plane) cudaFree(d_uv_plane);
    if (d_rgb_plane) cudaFree(d_rgb_plane);
    if (d_bgr_plane) cudaFree(d_bgr_plane);
  }
};

bool AllocateGpuMemory(TestFrame& frame) {
  size_t y_size = frame.width * frame.height;
  size_t uv_size = frame.width * frame.height / 2;

  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_y_plane, y_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_uv_plane, uv_size));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_rgb_plane, y_size * 3));
  CHECK_CUDA_RUNTIME(cudaMalloc(&frame.d_bgr_plane, y_size * 3));

  return true;
}

bool LoadImageAndConvertToNV12(const std::string& image_path, TestFrame& frame) {
  cv::Mat src_mat = cv::imread(image_path, cv::IMREAD_COLOR);
  if (src_mat.empty()) {
    std::cerr << "Failed to load image: " << image_path << std::endl;
    return false;
  }

  frame.width = src_mat.cols;
  frame.height = src_mat.rows;

  if (frame.height % 2 != 0 || frame.width % 2 != 0) {
    frame.height = (frame.height / 2) * 2;
    frame.width = (frame.width / 2) * 2;
    src_mat = src_mat(cv::Rect(0, 0, frame.width, frame.height));
  }

  std::cout << "Image loaded: " << frame.width << "x" << frame.height << std::endl;

  std::vector<uint8_t> bgr_buffer(src_mat.cols * src_mat.rows * 3);
  memcpy(bgr_buffer.data(), src_mat.data, bgr_buffer.size());

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);

  // according to libyuv demo
  std::vector<uint8_t> argb_buffer(frame.width * frame.height * 4);
  int                  argb_stride = frame.width * 4;
  // for RGB and BGR, stride[0] = cols * 3
  libyuv::RGB24ToARGB(bgr_buffer.data(), frame.width * 3, argb_buffer.data(), argb_stride, frame.width, frame.height);
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, frame.y_plane.data(), frame.width, 
                    frame.uv_plane.data(), frame.width, frame.width, frame.height);

  return true;
}

// 先转换 NV 再拷贝到 GPU, 等待转换
bool CopyToGpu(TestFrame& frame) {
  size_t y_size = frame.width * frame.height;
  size_t uv_size = frame.width * frame.height / 2;

  CHECK_CUDA_RUNTIME(cudaMemcpy(frame.d_y_plane, frame.y_plane.data(), y_size, cudaMemcpyHostToDevice));
  CHECK_CUDA_RUNTIME(cudaMemcpy(frame.d_uv_plane, frame.uv_plane.data(), uv_size, cudaMemcpyHostToDevice));

  return true;
}

bool TestNV12ToRGB24_CUDA(TestFrame& frame) {
  std::cout << "\n=== Testing NV12 -> RGB24 (CUDA) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  NV12ToRGB24Kernel<<<grid, block>>>(
      static_cast<const uint8_t*>(frame.d_y_plane), static_cast<const uint8_t*>(frame.d_uv_plane),
      static_cast<uint8_t*>(frame.d_rgb_plane), frame.width, frame.height, frame.width, frame.width, frame.width * 3);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  // 从 void* 到 vector<uint8_t> rgb_plane
  frame.rgb_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.rgb_plane.data(), frame.d_rgb_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, frame.rgb_plane.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  cv::imwrite("nv12_to_rgb24_cuda.png", bgr_mat);
  std::cout << "NV12 -> RGB24 (CUDA) result saved to: nv12_to_rgb24_cuda.png" << std::endl;

  return true;
}

bool TestNV12ToBGR24_CUDA(TestFrame& frame) {
  std::cout << "\n=== Testing NV12 -> BGR24 (CUDA) ===" << std::endl;

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  NV12ToBGR24Kernel<<<grid, block>>>(
      static_cast<const uint8_t*>(frame.d_y_plane), static_cast<const uint8_t*>(frame.d_uv_plane),
      static_cast<uint8_t*>(frame.d_bgr_plane), frame.width, frame.height, frame.width, frame.width, frame.width * 3);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.bgr_plane.data(), frame.d_bgr_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite("nv12_to_bgr24_cuda.png", bgr_mat);
  std::cout << "NV12 -> BGR24 (CUDA) result saved to: nv12_to_bgr24_cuda.png" << std::endl;

  return true;
}

bool TestRGB24ToBGR24_CUDA(TestFrame& frame) {
  std::cout << "\n=== Testing RGB24 -> BGR24 (CUDA) ===" << std::endl;

  if (frame.rgb_plane.empty()) {
    std::cerr << "RGB plane is empty, run NV12ToRGB24_CUDA first" << std::endl;
    return false;
  }

  // 从 vector<uint8_t> rgb_plane 到 void* d_rgb_plane
  // TODO: 前面 TestNV12ToRGB24_CUDA 是 d_rgb_plane 拷贝到了 rgb_plane
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.d_rgb_plane, frame.rgb_plane.data(), frame.width * frame.height * 3, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((frame.width + block.x - 1) / block.x, (frame.height + block.y - 1) / block.y);

  RGB24ToBGR24Kernel<<<grid, block>>>(static_cast<const uint8_t*>(frame.d_rgb_plane),
                                      static_cast<uint8_t*>(frame.d_bgr_plane), frame.width, frame.height,
                                      frame.width * 3);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  frame.bgr_plane.resize(frame.width * frame.height * 3);
  CHECK_CUDA_RUNTIME(
      cudaMemcpy(frame.bgr_plane.data(), frame.d_bgr_plane, frame.width * frame.height * 3, cudaMemcpyDeviceToHost));

  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3, frame.bgr_plane.data());
  cv::imwrite("rgb24_to_bgr24_cuda.png", bgr_mat);
  std::cout << "RGB24 -> BGR24 (CUDA) result saved to: rgb24_to_bgr24_cuda.png" << std::endl;

  return true;
}

bool TestWithLibyuvCPU(TestFrame& frame) {
  std::cout << "\n=== Testing with libyuv (CPU) for comparison ===" << std::endl;

  std::vector<uint8_t> cpu_rgb(frame.width * frame.height * 3);
  std::vector<uint8_t> cpu_bgr(frame.width * frame.height * 3);

  // actual: To RGB24
  int ret = libyuv::NV12ToRAW(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width, 
                            cpu_rgb.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRAW failed with error: " << ret << std::endl;
    return false;
  }

  // actual: To BGR24
  ret = libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width, 
                            cpu_bgr.data(), frame.width * 3, frame.width, frame.height);

  if (ret != 0) {
    std::cerr << "libyuv::NV12ToRGB24 failed with error: " << ret << std::endl;
    return false;
  }

  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3, cpu_rgb.data());
  cv::Mat bgr_mat;
  cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  cv::imwrite("nv12_to_rgb24_libyuv.png", bgr_mat);
  std::cout << "NV12 -> RGB24 (libyuv) result saved to: nv12_to_rgb24_libyuv.png" << std::endl;

  cv::Mat bgr_mat2(frame.height, frame.width, CV_8UC3, cpu_bgr.data());
  cv::imwrite("nv12_to_bgr24_libyuv.png", bgr_mat2);
  std::cout << "NV12 -> BGR24 (libyuv) result saved to: nv12_to_bgr24_libyuv.png" << std::endl;

  if (!frame.rgb_plane.empty()) {
    size_t diff_rgb = 0;
    for (size_t i = 0; i < frame.rgb_plane.size(); ++i) {
      if (std::abs(static_cast<int>(frame.rgb_plane[i]) - static_cast<int>(cpu_rgb[i])) > 1) {
        diff_rgb++;
      }
    }

    size_t total_pixels = frame.width * frame.height * 3;
    double diff_ratio_rgb = 100.0 * diff_rgb / total_pixels;

    std::cout << "NV12 -> RGB24 CUDA vs libyuv: " << diff_rgb << " / " << total_pixels << " pixels different ("
              << diff_ratio_rgb << "%)" << std::endl;
  }

  if (!frame.bgr_plane.empty()) {
    size_t diff_bgr = 0;
    for (size_t i = 0; i < frame.bgr_plane.size(); ++i) {
      if (std::abs(static_cast<int>(frame.bgr_plane[i]) - static_cast<int>(cpu_bgr[i])) > 1) {
        diff_bgr++;
      }
    }

    size_t total_pixels = frame.width * frame.height * 3;
    double diff_ratio_bgr = 100.0 * diff_bgr / total_pixels;

    std::cout << "NV12 -> BGR24 CUDA vs libyuv: " << diff_bgr << " / " << total_pixels << " pixels different ("
              << diff_ratio_bgr << "%)" << std::endl;
  }

  return true;
}

bool CreateUniformTestImage(int width, int height, uint8_t r_val, uint8_t g_val, uint8_t b_val, TestFrame& frame) {
  std::cout << "\n=== Creating uniform test image (R=" << (int)r_val 
            << ", G=" << (int)g_val << ", B=" << (int)b_val << ") ===" << std::endl;

  frame.width = width;
  frame.height = height;
  frame.fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;

  if (frame.height % 2 != 0 || frame.width % 2 != 0) {
    frame.height = (frame.height / 2) * 2;
    frame.width = (frame.width / 2) * 2;
  }

  cv::Mat src_mat(frame.height, frame.width, CV_8UC3);
  for (int y = 0; y < frame.height; ++y) {
    for (int x = 0; x < frame.width; ++x) {
      src_mat.at<cv::Vec3b>(y, x) = cv::Vec3b(b_val, g_val, r_val);
    }
  }

  cv::imwrite("uniform_test_original_bgr.png", src_mat);
  std::cout << "Original BGR image saved to: uniform_test_original_bgr.png" << std::endl;

  frame.y_plane.resize(frame.width * frame.height);
  frame.uv_plane.resize(frame.width * frame.height / 2);

  std::vector<uint8_t> bgr_buffer(frame.width * frame.height * 3);
  memcpy(bgr_buffer.data(), src_mat.data, bgr_buffer.size());

  std::vector<uint8_t> argb_buffer(frame.width * frame.height * 4);
  int                  argb_stride = frame.width * 4;
  libyuv::RGB24ToARGB(bgr_buffer.data(), frame.width * 3, argb_buffer.data(), argb_stride, frame.width, frame.height);
  libyuv::ARGBToNV12(argb_buffer.data(), argb_stride, frame.y_plane.data(), frame.width, 
                    frame.uv_plane.data(), frame.width, frame.width, frame.height);

  std::cout << "Image converted to NV12 format" << std::endl;
  return true;
}


/**
 * 检查经过核函数转换之后，每个通道的内存排列
 */
bool TestChannelConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n=== Testing channel consistency (Expected: R=" << (int)expected_r 
            << ", G=" << (int)expected_g << ", B=" << (int)expected_b << ") ===" << std::endl;

  if (frame.bgr_plane.empty()) {
    std::cerr << "BGR plane is empty, run TestNV12ToBGR24_CUDA first" << std::endl;
    return false;
  }

  int width = frame.width;
  int height = frame.height;

  size_t total_pixels = width * height;
  size_t b_errors = 0, g_errors = 0, r_errors = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      uint8_t b = frame.bgr_plane[idx + 0];
      uint8_t g = frame.bgr_plane[idx + 1];
      uint8_t r = frame.bgr_plane[idx + 2];

      if (std::abs(static_cast<int>(b) - static_cast<int>(expected_b)) > 1) b_errors++;
      if (std::abs(static_cast<int>(g) - static_cast<int>(expected_g)) > 1) g_errors++;
      if (std::abs(static_cast<int>(r) - static_cast<int>(expected_r)) > 1) r_errors++;
    }
  }

  // std::cout << "Channel consistency check:" << std::endl;
  // std::cout << "  B channel: " << b_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * b_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  G channel: " << g_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * g_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  R channel: " << r_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * r_errors / total_pixels) << "%)" << std::endl;

  std::cout << "\nBGR memory layout analysis:" << std::endl;
  std::cout << "  Memory[0] = B = " << (int)frame.bgr_plane[0] << " (expected: " << (int)expected_b << ")" << std::endl;
  std::cout << "  Memory[1] = G = " << (int)frame.bgr_plane[1] << " (expected: " << (int)expected_g << ")" << std::endl;
  std::cout << "  Memory[2] = R = " << (int)frame.bgr_plane[2] << " (expected: " << (int)expected_r << ")" << std::endl;

  bool b_match = (b_errors == 0);
  bool g_match = (g_errors == 0);
  bool r_match = (r_errors == 0);

  if (b_match && g_match && r_match) {
    std::cout << "\n[PASS] All channels match expected values!" << std::endl;
  } else {
    std::cout << "\n[FAIL] Channel mismatch detected!" << std::endl;
  }

  return (b_match && g_match && r_match);
}

bool TestOpenCVConversionConsistency(TestFrame& frame, uint8_t expected_r, uint8_t expected_g, uint8_t expected_b) {
  std::cout << "\n=== Testing OpenCV NV12 -> BGR24 conversion ===" << std::endl;

  std::vector<uint8_t> opencv_bgr(frame.width * frame.height * 3);
  
  cv::Mat rgb_mat(frame.height, frame.width, CV_8UC3);
  cv::Mat bgr_mat(frame.height, frame.width, CV_8UC3);

  libyuv::NV12ToRGB24(frame.y_plane.data(), frame.width, frame.uv_plane.data(), frame.width,
                      rgb_mat.data, frame.width * 3, frame.width, frame.height);
                      
  // cv::cvtColor(rgb_mat, bgr_mat, cv::COLOR_RGB2BGR);
  // memcpy(opencv_bgr.data(), bgr_mat.data, opencv_bgr.size());
  // cv::imwrite("nv12_to_bgr24_opencv.png", bgr_mat);

  // note: 根据之前 libyuv_demo 的分析结果， 不再需要 cvtColor
  // 根据实验结果，证实如此
  memcpy(opencv_bgr.data(), rgb_mat.data, opencv_bgr.size());
  cv::imwrite("nv12_to_bgr24_opencv.png", rgb_mat);
  
  std::cout << "OpenCV conversion result saved to: nv12_to_bgr24_opencv.png" << std::endl;

  int width = frame.width;
  int height = frame.height;
  size_t total_pixels = width * height;
  size_t b_errors = 0, g_errors = 0, r_errors = 0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int idx = (y * width + x) * 3;
      uint8_t b = opencv_bgr[idx + 0];
      uint8_t g = opencv_bgr[idx + 1];
      uint8_t r = opencv_bgr[idx + 2];

      if (std::abs(static_cast<int>(b) - static_cast<int>(expected_b)) > 1) b_errors++;
      if (std::abs(static_cast<int>(g) - static_cast<int>(expected_g)) > 1) g_errors++;
      if (std::abs(static_cast<int>(r) - static_cast<int>(expected_r)) > 1) r_errors++;
    }
  }

  // std::cout << "OpenCV channel consistency check:" << std::endl;
  // std::cout << "  B channel: " << b_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * b_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  G channel: " << g_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * g_errors / total_pixels) << "%)" << std::endl;
  // std::cout << "  R channel: " << r_errors << " / " << total_pixels << " pixels different ("
  //           << (100.0 * r_errors / total_pixels) << "%)" << std::endl;

  std::cout << "\nOpenCV BGR memory layout:" << std::endl;
  std::cout << "  Memory[0] = B = " << (int)opencv_bgr[0] << " (expected: " << (int)expected_b << ")" << std::endl;
  std::cout << "  Memory[1] = G = " << (int)opencv_bgr[1] << " (expected: " << (int)expected_g << ")" << std::endl;
  std::cout << "  Memory[2] = R = " << (int)opencv_bgr[2] << " (expected: " << (int)expected_r << ")" << std::endl;

  return (b_errors == 0 && g_errors == 0 && r_errors == 0);
}

int main(int argc, char** argv) {
  std::string image_path = (argc > 1) ? argv[1] : DEFAULT_IMAGE_PATH;

  std::cout << "========================================" << std::endl;
  std::cout << "  CUDA Image Format Conversion Demo   " << std::endl;
  std::cout << "========================================" << std::endl;

  int         deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  if (err != cudaSuccess || deviceCount == 0) {
    std::cerr << "No CUDA devices found!" << std::endl;
    return -1;
  }

  std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Using device 0: " << prop.name << std::endl;

  TestFrame frame;
  frame.fmt = DataFormat::PIXEL_FORMAT_YUV420_NV12;

  std::cout << "\nLoading image: " << image_path << std::endl;
  if (!LoadImageAndConvertToNV12(image_path, frame)) {
    std::cerr << "Failed to load image and convert to NV12" << std::endl;
    return -1;
  }

  std::cout << "Allocating GPU memory..." << std::endl;
  if (!AllocateGpuMemory(frame)) {
    std::cerr << "Failed to allocate GPU memory" << std::endl;
    return -1;
  }

  std::cout << "Copying data to GPU..." << std::endl;
  CopyToGpu(frame);

  TestNV12ToRGB24_CUDA(frame);
  TestNV12ToBGR24_CUDA(frame);
  TestRGB24ToBGR24_CUDA(frame);
  TestWithLibyuvCPU(frame);

  std::cout << "\n\n";
  std::cout << "########################################" << std::endl;
  std::cout << "#  Channel Consistency Test (R=G=B)  #" << std::endl;
  std::cout << "########################################" << std::endl;

  TestFrame uniform_frame;
  const int test_width = 640;
  const int test_height = 480;
  const uint8_t test_r = 10;
  const uint8_t test_g = 128;
  const uint8_t test_b = 242;

  if (!CreateUniformTestImage(test_width, test_height, test_r, test_g, test_b, uniform_frame)) {
    std::cerr << "Failed to create uniform test image" << std::endl;
    return -1;
  } 
  if (!AllocateGpuMemory(uniform_frame)) {
    std::cerr << "Failed to allocate GPU memory for uniform frame" << std::endl;
  } else {
    CopyToGpu(uniform_frame);
    TestNV12ToBGR24_CUDA(uniform_frame);
    TestChannelConsistency(uniform_frame, test_r, test_g, test_b);
    TestOpenCVConversionConsistency(uniform_frame, test_r, test_g, test_b);
  }

  std::cout << "\n========================================" << std::endl;
  std::cout << "  Demo completed successfully!        " << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}
