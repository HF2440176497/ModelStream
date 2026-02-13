
#include "cnstream_logging.hpp"

#include "data_source_param.hpp"  // DevContext, DataFormat
#include "data_handler_util.hpp"

namespace cnstream {

int SourceRender::Process(std::shared_ptr<FrameInfo> frame_info, DecodeFrame *decode_frame, uint64_t frame_id) {
  DataFramePtr frame = frame_info->collection.Get<DataFramePtr>(kDataFrameTag);
  if (!frame || !decode_frame) {
    LOGF(SOURCE) << "SourceRender::Process: frame or decode_frame is NULL";
    return -1;
  }
  if (!decode_frame->valid) return -1;
  // 1.1 common operation
  frame->frame_id = frame_id;
  frame->width = decode_frame->width;
  frame->height = decode_frame->height;
  if (decode_frame->buf_ref) {
    frame->deAllocator_ = std::make_unique<Deallocator>(decode_frame->buf_ref.release());  // decode_frame 中的内存转移到 frame deAllocator_, 后续由 frame 管理
    decode_frame->buf_ref = nullptr;
  }
  // 1.2 ctx 保存目标 dev info
  // 在解码完成之后，DataSource 的模块参数不再使用，而是直接使用已得到的编码帧 decode_frame
  frame->ctx = DevContext(decode_frame->dev_type, decode_frame->device_id);

  // note: 固定为 RGB24 格式
  frame->fmt = DataFormat::PIXEL_FORMAT_RGB24;
  for (int i = 0; i < frame->GetPlanes(); ++i) {
    if (i == 0) {
      frame->stride[i] = frame->width * 3;
    }
  }
  // 2. 创建 memop 来拷贝图像内存
  frame->CopyToSyncMem(decode_frame);
  return 0;
}

}