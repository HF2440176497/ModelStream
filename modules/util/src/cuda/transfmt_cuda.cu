

#include "cuda/transfmt_cuda.cuh"

#include <cstring>


namespace cnstream {

#define CHECK_NPP(op) __check_npp((op), #op, __FILE__, __LINE__)

static std::string nppGetStatusString(NppStatus code) {
    return "NPP error code: " + std::to_string(code);
}

static bool __check_npp(NppStatus code, const char* op, const char* file, int line) {
  if (code != NPP_SUCCESS) {
    printf("check_npp error %s:%d  %s failed. \n  code = %d, message = %s\n",
           file, line, op, code, nppGetStatusString(code).c_str());
    return false;
  }
  return true;
}

int NppNV12ToRGB24(void* dst, int width, int height, const void* y_plane, const void* uv_plane, cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(y_plane),
    static_cast<const Npp8u*>(uv_plane),
  };
  int aSrcStep = width;

  Npp8u* pDst = static_cast<Npp8u*>(dst);
  int nDstStep = width * 3;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  status = nppiNV12ToRGB_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    oSizeROI,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppNV12ToBGR24(void* dst, int width, int height, const void* y_plane, const void* uv_plane, cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  const Npp8u* aSrc[2] = {
    static_cast<const Npp8u*>(y_plane),
    static_cast<const Npp8u*>(uv_plane),
  };
  int aSrcStep = width;

  Npp8u* pDst = static_cast<Npp8u*>(dst);
  int nDstStep = width * 3;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  status = nppiNV12ToBGR_8u_P2C3R_Ctx(
    aSrc, aSrcStep,
    pDst, nDstStep,
    oSizeROI,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppRGB24ToBGR24(void* dst, int width, int height, const void* src, cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  int nStep = width * 3;

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
    static_cast<const Npp8u*>(src),
    nStep,
    static_cast<Npp8u*>(dst),
    nStep,
    oSizeROI,
    aDstOrder,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

int NppBGR24ToRGB24(void* dst, int width, int height, const void* src, cudaStream_t stream) {
  NppStreamContext npp_stream_ctx;
  NppStatus status = nppGetStreamContext(&npp_stream_ctx);
  CHECK_NPP(status);
  npp_stream_ctx.hStream = stream;

  NppiSize oSizeROI;
  oSizeROI.width   = width;
  oSizeROI.height  = height;

  int nStep = width * 3;

  int aDstOrder[3] = { 2, 1, 0 };
  status = nppiSwapChannels_8u_C3R_Ctx(
    static_cast<const Npp8u*>(src),
    nStep,
    static_cast<Npp8u*>(dst),
    nStep,
    oSizeROI,
    aDstOrder,
    npp_stream_ctx
  );
  CHECK_NPP(status);

  CHECK_CUDA_RUNTIME(cudaGetLastError());
  CHECK_CUDA_RUNTIME(cudaDeviceSynchronize());

  return 0;
}

}  // namespace cnstream
