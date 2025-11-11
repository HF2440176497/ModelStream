

#ifndef CNSTREAM_FRAME_VA_HPP_
#define CNSTREAM_FRAME_VA_HPP_


#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "cnstream_common.hpp"
#include "cnstream_frame.hpp"


namespace cnstream {

inline constexpr uint32_t CN_MAX_PLANES = 6;

/**
 * @enum CNDataFormat
 *
 * @brief Enumeration variables describling the pixel format of the data in CNDataFrame.
 */
enum class CNDataFormat {
  CN_INVALID = -1,                 /*!< This frame is invalid. */
  CN_PIXEL_FORMAT_YUV420_NV21 = 0, /*!< This frame is in the YUV420SP(NV21) format. */
  CN_PIXEL_FORMAT_YUV420_NV12,     /*!< This frame is in the YUV420sp(NV12) format. */
  CN_PIXEL_FORMAT_BGR24,           /*!< This frame is in the BGR24 format. */
  CN_PIXEL_FORMAT_RGB24,           /*!< This frame is in the RGB24 format. */
  CN_PIXEL_FORMAT_ARGB32,          /*!< This frame is in the ARGB32 format. */
  CN_PIXEL_FORMAT_ABGR32,          /*!< This frame is in the ABGR32 format. */
  CN_PIXEL_FORMAT_RGBA32,          /*!< This frame is in the RGBA32 format. */
  CN_PIXEL_FORMAT_BGRA32           /*!< This frame is in the BGRA32 format. */
};

/**
 * @struct DevContext
 *
 * @brief DevContext is a structure holding the information that CNDataFrame data is allocated by CPU or MLU.
 */
struct DevContext {
  enum class DevType {
    INVALID = -1,                /*!< Invalid device type. */
    CPU = 0,                     /*!< The data is allocated by CPU. */
    MLU = 1,                     /*!< The data is allocated by MLU. */
  } dev_type = DevType::INVALID; /*!< Device type. The default value is ``INVALID``.*/
  int dev_id = 0;                /*!< Ordinal device ID. */
  int ddr_channel = 0;           /*!< Ordinal channel ID for MLU. The value should be in the range [0, 4). */
};


/**
 * @brief Gets image plane number by a specified image format.
 * 表示数量，范围为自然数
 * @retval 0: Unsupported image format.
 * @retval >0: Image plane number.
 */
inline int CNGetPlanes(CNDataFormat fmt) {
  switch (fmt) {
    case CNDataFormat::CN_PIXEL_FORMAT_BGR24:
    case CNDataFormat::CN_PIXEL_FORMAT_RGB24:
      return 1;
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV12:
    case CNDataFormat::CN_PIXEL_FORMAT_YUV420_NV21:
      return 2;
    default:
      return 0;
  }
  return 0;
}


/**
 * @class CNDataFrame
 * @brief CNDataFrame is a class holding a data frame and the frame description.
 * @todo: 未来支持统一内存管理
 * 在外使用 shared_ptr 管理
 */
class CNDataFrame : public NonCopyable {
 public:
  /**
   * @brief Constructs an object.
   *
   * @return No return value.
   */
  CNDataFrame() = default;
  /**
   * @brief Destructs an object.
   *
   * @return No return value.
   */
  ~CNDataFrame() = default;
  /**
   * @brief Gets plane count for a specified frame.
   *
   * @return Returns the plane count of this frame.
   */
  int GetPlanes() const { return CNGetPlanes(fmt); }
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

 public:
  /**
   * @brief Converts data to the BGR format.
   *
   * @return Returns data with OpenCV mat type.
   *
   * @note This function is called after CNDataFrame::CopyToSyncMem() is invoked.
   */
  cv::Mat ImageBGR();
  /**
   * @brief Checks whether there is BGR image stored.
   *
   * @return Returns true if has BGR image, otherwise returns false.
   */
  bool HasBGRImage() {
    std::lock_guard<std::mutex> lk(mtx);
    if (bgr_mat.empty()) return false;
    return true;
  }

  std::shared_ptr<void> cpu_data = nullptr;            /*!< A shared pointer to the CPU data. */
  std::shared_ptr<void> mlu_data = nullptr;            /*!< A shared pointer to the MLU data. */
  uint64_t frame_id = -1;                              /*!< The frame index that incremented from 0. */

  CNDataFormat fmt;                                         /*!< The format of the frame. */
  int width;                                                /*!< The width of the frame. */
  int height;                                               /*!< The height of the frame. */
  int stride[CN_MAX_PLANES];                                /*!< The strides of the frame. */
  DevContext ctx;                                           /*!< The device context of SOURCE data (ptr_mlu/ptr_cpu). */
  std::atomic<int> dst_device_id{-1};                       /*!< The device context of SyncedMemory. */

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::mutex mtx;
  cv::Mat bgr_mat; /*!< A Mat stores BGR image. */

};  // class CNDataFrame


using CNDataFramePtr = std::shared_ptr<CNDataFrame>;
// using CNInferObjsPtr = std::shared_ptr<CNInferObjs>;
// using CNObjsVec = std::vector<std::shared_ptr<CNInferObject>>;
// using CNInferDataPtr = std::shared_ptr<CNInferData>;


static constexpr char kCNDataFrameTag[] = "CNDataFrame"; /*!< value type in CNFrameInfo::Collection : CNDataFramePtr. */
static constexpr char kCNInferObjsTag[] = "CNInferObjs"; /*!< value type in CNFrameInfo::Collection : CNInferObjsPtr. */
static constexpr char kCNInferDataTag[] = "CNInferData"; /*!< value type in CNFrameInfo::Collection : CNInferDataPtr. */


}  // namespace cnstream
#endif