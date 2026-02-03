

#include <mutex>
#include <memory>

#include "cnstream_frame_va.hpp"


namespace cnstream {


size_t DataFrame::GetPlaneBytes(int plane_idx) const {
  if (plane_idx < 0 || plane_idx >= GetPlanes()) return 0;
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return height * stride[0] * 3;
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
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

size_t DataFrame::GetBytes() const {
  size_t bytes = 0;
  for (int i = 0; i < GetPlanes(); ++i) {
    bytes += GetPlaneBytes(i);
  }
  return bytes;
}


}  // namespace cnstream
