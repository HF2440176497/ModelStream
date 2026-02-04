

#ifndef CNSTREAM_FRAME_VA_HPP_
#define CNSTREAM_FRAME_VA_HPP_


#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cnstream_common.hpp"
#include "cnstream_frame.hpp"


namespace cnstream {

inline constexpr uint32_t CN_MAX_PLANES = 6;

/**
 * @enum DataFormat
 *
 * @brief Enumeration variables describling the pixel format of the data in DataFrame.
 */
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

enum class DevType {
  INVALID = -1,                /*!< Invalid device type. */
  CPU = 0,                     /*!< The data is allocated by CPU. */
  CUDA = 1,                    /*!< The data is allocated by CUDA. */
};

/**
 * @struct DevContext
 *
 * @brief DevContext is a structure holding the information that DataFrame data is allocated by CPU or MLU.
 */
struct DevContext {
  DevType dev_type = DevType::INVALID; 
  int dev_id = 0;
};


/**
 * @brief Gets image plane number by a specified image format.
 * 表示数量，范围为自然数
 * @retval 0: Unsupported image format.
 * @retval >0: Image plane number.
 */
inline int FormatPlanes(DataFormat fmt) {
  switch (fmt) {
    case DataFormat::PIXEL_FORMAT_BGR24:
    case DataFormat::PIXEL_FORMAT_RGB24:
      return 1;
    case DataFormat::PIXEL_FORMAT_YUV420_NV12:
    case DataFormat::PIXEL_FORMAT_YUV420_NV21:
      return 2;
    default:
      return 0;
  }
  return 0;
}


/**
 * @class DataFrame
 * @brief DataFrame is a class holding a data frame and the frame description.
 * @todo: 未来支持统一内存管理
 * 在外使用 shared_ptr 管理
 */
class DataFrame : public NonCopyable {
 public:
  /**
   * @brief Constructs an object.
   *
   * @return No return value.
   */
  DataFrame() = default;
  /**
   * @brief Destructs an object.
   *
   * @return No return value.
   */
  ~DataFrame() = default;

  /**
   * @brief Gets plane count for a specified frame.
   *
   * @return Returns the plane count of this frame.
   */
  int GetPlanes() const { return FormatPlanes(fmt); }
  /**
   * @brief Gets the number of bytes in a specified plane.
   *
   * @param[in] plane_idx The index of the plane. The index increments from 0.
   *
   * @return Returns the number of bytes in the plane.
   */
  size_t GetPlaneBytes(int plane_idx) const;
  /**
   * @brief Gets the number of bytes in a frame.
   *
   * @return Returns the number of bytes in a frame.
   */
  size_t GetBytes() const;

  /**
   * @brief 复制数据到同步内存
   */
  void CopyToSyncMem(DecodeFrame* decode_frame);

  /**
   * @brief 创建MemOp
   * @return 返回创建的MemOp实例，如果不支持该设备类型则返回nullptr
   */
  std::unique_ptr<MemOp> CreateMemOp();

  /**
   * @brief Converts data to the BGR format.
   *
   * @return Returns data with OpenCV mat type.
   *
   * @note This function is called after CNDataFrame::CopyToSyncMem() is invoked.
   */
  cv::Mat GetImage();
  /**
   * @brief Checks whether there is BGR image stored.
   *
   * @return Returns true if has BGR image, otherwise returns false.
   */
  bool HasImage() {
    std::lock_guard<std::mutex> lk(mtx);
    if (mat_.empty()) return false;
    return true;
  }

public:
  MemoryBufferCollection mem_manager_;  // 内存分配集合
  std::unique_ptr<CNSyncedMemory> data[CN_MAX_PLANES];
  uint64_t frame_id = -1;                              /*!< The frame index that incremented from 0. */

  DataFormat fmt;                                         /*!< The format of the frame. */
  int width;                                                /*!< The width of the frame. */
  int height;                                               /*!< The height of the frame. */
  int stride[CN_MAX_PLANES];                                /*!< The strides of the frame. */
  DevContext ctx;                                           /*!< The device context of SOURCE data (ptr_mlu/ptr_cpu). */

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::mutex mtx;
  cv::Mat mat_;  /*!< A Mat stores BGR image. */
};  // class DataFrame


using DataFramePtr = std::shared_ptr<DataFrame>;
// using InferObjsPtr = std::shared_ptr<InferObjs>;
// using InferObjsVec = std::vector<std::shared_ptr<InferObject>>;
// using InferDataPtr = std::shared_ptr<InferData>;


inline constexpr char kCNDataFrameTag[] = "CNDataFrame"; /*!< value type in CNFrameInfo::Collection : CNDataFramePtr. */
inline constexpr char kCNInferObjsTag[] = "CNInferObjs"; /*!< value type in CNFrameInfo::Collection : CNInferObjsPtr. */
inline constexpr char kCNInferDataTag[] = "CNInferData"; /*!< value type in CNFrameInfo::Collection : CNInferDataPtr. */


}  // namespace cnstream
#endif