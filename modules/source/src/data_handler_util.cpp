
#include "cnstream_logging.hpp"

#include "data_source_param.hpp"  // DevContext, DataFormat
#include "data_handler_util.hpp"

namespace cnstream {

/**
 * OnDecodeFrame 同步调用
 */
int SourceRender::Process(std::shared_ptr<FrameInfo> frame_info, DecodeFrame *decode_frame, uint64_t frame_id) {
  DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame || !decode_frame) {
    LOGF(SOURCE) << "SourceRender::Process: frame or decode_frame is NULL";
    return -1;
  }
  if (!decode_frame->valid) return -1;
  frame->frame_id_ = frame_id;
  frame->width_ = decode_frame->width;
  frame->height_ = decode_frame->height;
  if (decode_frame->buf_ref) {
    frame->deAllocator_ = std::make_unique<Deallocator>(decode_frame->buf_ref.release());
    decode_frame->buf_ref = nullptr;
  }
  frame->ctx_ = DevContext(decode_frame->dev_type, decode_frame->device_id);

  frame->fmt_ = DataFormat::PIXEL_FORMAT_RGB24;
  for (int i = 0; i < frame->GetPlanes(); ++i) {
    if (i == 0) {
      frame->stride_[i] = frame->width_ * 3;
    }
  }
  frame->CopyToSyncMem(decode_frame);
  return 0;
}

}