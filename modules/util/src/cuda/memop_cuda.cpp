// cuda_memop.cpp

#include "memop_factory.hpp"

#include "cnstream_logging.hpp"
#include "cuda/memop_cuda.hpp"
#include "cuda/cuda_check.hpp"
#include "cuda/cnstream_sysncmem_cuda.hpp"
#include "cuda/transfmt_cuda.cuh"

namespace cnstream {

static bool RegisterCudaMemOp() {
  auto& factory = MemOpFactory::Instance();
  bool result = true;
  result &= factory.RegisterMemOpCreator(DevType::CUDA,
    [](int dev_id) {
      return std::make_unique<CudaMemOp>(dev_id);
    });
  return result;
}

static bool cuda_memops_registered = RegisterCudaMemOp();

CudaMemOp::CudaMemOp() {}

CudaMemOp::~CudaMemOp() {}

std::unique_ptr<CNSyncedMemory> CudaMemOp::CreateSyncedMemory(size_t size) {
  return std::make_unique<CNSyncedMemoryCuda>(size, device_id_);
}

std::shared_ptr<void> CudaMemOp::Allocate(size_t bytes) {
  bytes = RoundUpSize(bytes);
  CudaDeviceGuard guard(device_id_);
  return cnCudaMemAlloc(bytes, device_id_);
}

void CudaMemOp::Copy(void* dst, const void* src, size_t size) {
  CudaDeviceGuard guard(device_id_);
  CHECK_CUDA_RUNTIME(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
}

void CudaMemOp::SetData(CNSyncedMemory* mem, void* data) {
  auto cuda_mem = dynamic_cast<CNSyncedMemoryCuda*>(mem);
  if (!cuda_mem) {
    throw std::runtime_error("CudaMemOp: mem is not CNSyncedMemoryCuda");
  }
  CudaDeviceGuard guard(device_id_);
  cuda_mem->SetCudaData(data);
}

/**
 * @brief 使用 CUDA, 将解码帧转换为 dst_fmt 格式
 */
int CudaMemOp::ConvertImageFormat(void* dst, DataFormat dst_fmt, const DecodeFrame* src_frame) {
  int width = src_frame->width;
  int height = src_frame->height;
  if (dst_fmt != DataFormat::PIXEL_FORMAT_BGR24 &&
      dst_fmt != DataFormat::PIXEL_FORMAT_RGB24) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
               << static_cast<int>(dst_fmt);
    return -1;
  }
  DataFormat src_fmt = src_frame->fmt;

  // RGB or BGR 只需要拷贝 plane[0]
  if (dst_fmt == src_fmt) {
    LOGW(CORE) << "CudaMemOp::ConvertImageFormat: Source format is same as destination format";
    Copy(dst, src_frame->plane[0], width * height * 3);
    return 0;
  }
  size_t dst_stride = width * 3;
  int ret = 0;

  switch (src_fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppRGB24ToBGR24(dst, width, height, src_frame->plane[0]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source BGR24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_RGB24: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppBGR24ToRGB24(dst, width, height, src_frame->plane[0]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source RGB24";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV12: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppNV12ToRGB24(dst, width, height, src_frame->plane[0], src_frame->plane[1]);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppNV12ToBGR24(dst, width, height, src_frame->plane[0], src_frame->plane[1]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV12";
        return -1;
      }
      break;
    }
    case DataFormat::PIXEL_FORMAT_YUV420_NV21: {
      if (dst_fmt == DataFormat::PIXEL_FORMAT_RGB24) {
        ret = NppNV21ToRGB24(dst, width, height, src_frame->plane[0], src_frame->plane[1]);
      } else if (dst_fmt == DataFormat::PIXEL_FORMAT_BGR24) {
        ret = NppNV21ToBGR24(dst, width, height, src_frame->plane[0], src_frame->plane[1]);
      } else {
        LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported destination format " 
                   << static_cast<int>(dst_fmt) << " for source NV21";
        return -1;
      }
      break;
    }
    default:
      LOGE(CORE) << "CudaMemOp::ConvertImageFormat: Unsupported source format " 
                 << static_cast<int>(src_fmt);
      return -1;
  }
  if (ret != 0) {
    LOGE(CORE) << "CudaMemOp::ConvertImageFormat: libyuv conversion failed with error code: " << ret;
    return ret;
  }
  return 0;
}


}  // namespace cnstream