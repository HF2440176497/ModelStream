

#include <mutex>
#include <memory>

#include "cnstream_frame_va.hpp"


namespace cnstream {

/*
namespace color_cvt {

static
cv::Mat BGRToBGR(const CNDataFrame& frame) {
  const cv::Mat bgr(frame.height, frame.stride[0], CV_8UC3, const_cast<void*>(frame.data[0]->GetCpuData()));
  return bgr(cv::Rect(0, 0, frame.width, frame.height)).clone();
}

static
cv::Mat RGBToBGR(const CNDataFrame& frame) {
  const cv::Mat rgb(frame.height, frame.stride[0], CV_8UC3, const_cast<void*>(frame.data[0]->GetCpuData()));
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
  return bgr(cv::Rect(0, 0, frame.width, frame.height)).clone();
}

static
cv::Mat YUV420SPToBGR(const CNDataFrame& frame, bool nv21) {
  const uint8_t* y_plane = reinterpret_cast<const uint8_t*>(frame.data[0]->GetCpuData());
  const uint8_t* uv_plane = reinterpret_cast<const uint8_t*>(frame.data[1]->GetCpuData());
  int width = frame.width;
  int height = frame.height;
  if (width <= 0 || height <= 1) {
    LOGF(FRAME) << "Invalid width or height, width = " << width << ", height = " << height;
  }
  height = height & (~static_cast<int>(1));

  int y_stride = frame.stride[0];
  int uv_stride = frame.stride[1];
  cv::Mat bgr(height, width, CV_8UC3);
  uint8_t* dst_bgr24 = bgr.data;
  int dst_stride = width * 3;
  if (nv21)
    libyuv::NV21ToRGB24Matrix(y_plane, y_stride, uv_plane, uv_stride,
                              dst_bgr24, dst_stride, &libyuv::kYvuH709Constants, width, height);
  else
    libyuv::NV12ToRGB24Matrix(y_plane, y_stride, uv_plane, uv_stride,
                              dst_bgr24, dst_stride, &libyuv::kYvuH709Constants, width, height);
  return bgr;
}


static inline
cv::Mat NV12ToBGR(const CNDataFrame& frame) {
  return YUV420SPToBGR(frame, false);
}

static inline
cv::Mat NV21ToBGR(const CNDataFrame& frame) {
  return YUV420SPToBGR(frame, true);
}

static inline
cv::Mat FrameToImageBGR(const CNDataFrame& frame) {
  switch (frame.fmt) {
    case CNDataFormat::CN_PIXEL_FORMAT_BGR24:
      return BGRToBGR(frame);
    case CNDataFormat::CN_PIXEL_FORMAT_RGB24:
      return RGBToBGR(frame);
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV12:
      return NV12ToBGR(frame);
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV21:
      return NV21ToBGR(frame);
    default:
      LOGF(FRAME) << "Unsupported pixel format. fmt[" << static_cast<int>(frame.fmt) << "]";
  }
  // never be here
  return cv::Mat();
}
}  // namespace color_cvt
*/

/*
size_t CNDataFrame::GetPlaneBytes(int plane_idx) const {
  if (plane_idx < 0 || plane_idx >= GetPlanes()) return 0;
  switch (fmt) {
    case CNDataFormat::CN_PIXEL_FORMAT_BGR24:
    case CNDataFormat::CN_PIXEL_FORMAT_RGB24:
      return height * stride[0] * 3;
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV12:
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV21:
      if (0 == plane_idx)
        return height * stride[0];
      else if (1 == plane_idx)
        return std::ceil(1.0 * height * stride[1] / 2);
      else
        LOGF(FRAME) << "plane index wrong.";
    default:
      return 0;
  }
  return 0;
}

size_t CNDataFrame::GetBytes() const {
  size_t bytes = 0;
  for (int i = 0; i < GetPlanes(); ++i) {
    bytes += GetPlaneBytes(i);
  }
  return bytes;
}
*/

cv::Mat CNDataFrame::ImageBGR() {
  std::lock_guard<std::mutex> lk(mtx);
  if (!bgr_mat.empty()) {
    return bgr_mat;
  }
  // bgr_mat = color_cvt::FrameToImageBGR(*this);
  return bgr_mat;
}

}  // namespace cnstream
