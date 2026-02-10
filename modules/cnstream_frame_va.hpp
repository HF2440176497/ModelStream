

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