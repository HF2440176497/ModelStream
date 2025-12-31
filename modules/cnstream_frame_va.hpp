

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
#include "util/cnstream_allocator.hpp"


namespace cnstream {

/**
 * @class CNDataFrame
 * @brief CNDataFrame is a class holding a data frame and the frame description.
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

  /**
   * @brief 复制数据到同步内存
   */
  void CopyToSyncMem(DecodeFrame* decode_frame);

  /**
   * @brief 创建MemOp
   * @return 返回创建的MemOp实例，如果不支持该设备类型则返回nullptr
   */
  std::unique_ptr<MemOp> CreateMemOp();
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
  MemoryBufferCollection mem_manager_;  // 内存分配集合
  std::unique_ptr<CNSyncedMemory> data[CN_MAX_PLANES];
  uint64_t frame_id = -1;                              /*!< The frame index that incremented from 0. */

  CNDataFormat fmt;                                         /*!< The format of the frame. */
  int width;                                                /*!< The width of the frame. */
  int height;                                               /*!< The height of the frame. */
  int stride[CN_MAX_PLANES];                                /*!< The strides of the frame. 来自 Encode 阶段 */
  DevContext ctx;                                           /*!< The device context of SOURCE data */
  std::unique_ptr<IDataDeallocator> deAllocator_ = nullptr; /* 内存复用器 */

#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  std::mutex mtx;
  cv::Mat mat_; /*!< A Mat stores BGR image. */

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