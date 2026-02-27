

#ifndef CNSTREAM_FRAME_VA_HPP_
#define CNSTREAM_FRAME_VA_HPP_


#include <memory>
#include <mutex>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <libyuv/convert.h>

#include "cnstream_common.hpp"
#include "cnstream_frame.hpp"

// ---- util
#include "memop.hpp"
#include "cnstream_syncmem.hpp"

namespace cnstream {


/**
 * @class DataFrame
 * @brief DataFrame is a class holding a data frame and the frame description.
 * @todo: 未来支持统一内存管理
 * 在外使用 shared_ptr 管理
 */
class DataFrame : public NonCopyable {
 public:
  DataFrame() {
    for (int i = 0; i < CN_MAX_PLANES; ++i) {
      data[i] = nullptr;
    }
    for (int i = 0; i < CN_MAX_PLANES; ++i) {
      stride[i] = 0;
    }
  };
  ~DataFrame() = default;

  int GetPlanes() const { return FormatPlanes(fmt_); }

  size_t GetPlaneBytes(int plane_idx) const;

  size_t GetBytes() const;

  void CopyToSyncMem(DecodeFrame* decode_frame);

  std::unique_ptr<MemOp> CreateMemOp();

  cv::Mat GetImage();

  bool HasImage() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (mat_.empty()) return false;
    return true;
  }

 public:
  uint64_t GetFrameId() const { return frame_id_; }
  DataFormat GetFmt() const { return fmt_; }
  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  int GetStride(int plane_idx) const { return stride_[plane_idx]; }
  const DevContext& GetCtx() const { return ctx_; }

  MemoryBufferCollection mem_manager_;
  std::unique_ptr<CNSyncedMemory> data[CN_MAX_PLANES];
  std::unique_ptr<IDataDeallocator> deAllocator_ = nullptr;
  
#ifdef UNIT_TEST
 public:
#else
 private:
#endif
  mutable std::mutex mtx_;
  cv::Mat mat_;

  uint64_t frame_id_ = -1;
  DataFormat fmt_ = DataFormat::INVALID;
  int width_ = 0;
  int height_ = 0;
  int stride_[CN_MAX_PLANES];
  DevContext ctx_;

  friend class SourceRender;
};  // class DataFrame


using DataFramePtr = std::shared_ptr<DataFrame>;
// using InferObjsPtr = std::shared_ptr<InferObjs>;
// using InferObjsVec = std::vector<std::shared_ptr<InferObject>>;
// using InferDataPtr = std::shared_ptr<InferData>;


inline constexpr char kDataFrameTag[] = "DataFrame"; /*!< value type in FrameInfo::Collection : DataFramePtr. */
inline constexpr char kInferObjsTag[] = "InferObjs"; /*!< value type in FrameInfo::Collection : InferObjsPtr. */
inline constexpr char kInferDataTag[] = "InferData"; /*!< value type in FrameInfo::Collection : InferDataPtr. */


}  // namespace cnstream
#endif