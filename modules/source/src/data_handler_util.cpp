
#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"

#include "util/memop.hpp"
#include "util/memop_factory.hpp"


int SourceRender::Process(std::shared_ptr<CNFrameInfo> frame_info, DecodeFrame *decode_frame, uint64_t frame_id,
                          const DataSourceParam &param_) {
  CNDataFramePtr frame = frame_info->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
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
    IDecBufRef *ptr = decode_frame->buf_ref.release();
    frame->deAllocator_.reset(new Deallocator(ptr));  // 转移到 deAllocator_ 指向
  }
  // 1.2 ctx 保存目标 dev info
  // 在解码完成之后，DataSource 的模块参数不再使用，而是直接使用已得到的编码帧 decode_frame
  frame->ctx = DevContext(decode_frame->dev_type, decode_frame->device_id);
  frame->fmt = CNDataFormat::CN_PIXEL_FORMAT_BGR24;  // 统一转换为 BGR24 格式
  // 2. 创建 memop 来拷贝图像内存
  frame->CopyToSyncMem(decode_frame);
  return 0;
}