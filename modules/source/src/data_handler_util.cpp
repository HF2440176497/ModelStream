
#include "cnstream_logging.hpp"
#include "data_handler_util.hpp"

#include "util/memop.hpp"
#include "util/memop_factory.hpp"


int SourceRender::Process(std::shared_ptr<CNFrameInfo> frame_info, DecodeFrame *decode_frame, uint64_t frame_id,
                          const DataSourceParam &param_) {
  CNDataFramePtr dataframe = frame_info->collection.Get<CNDataFramePtr>(kCNDataFrameTag);
  if (!dataframe || !decode_frame) {
    LOGF(SOURCE) << "SourceRender::Process: dataframe or decode_frame is NULL";
    return -1;
  }
  if (!decode_frame->valid) return -1;
  // 1. common operation
  dataframe->frame_id = frame_id;
  dataframe->width = decode_frame->width;
  dataframe->height = decode_frame->height;
  if (decode_frame->buf_ref) {
    IDecBufRef *ptr = decode_frame->buf_ref.release();
    dataframe->deAllocator_.reset(new Deallocator(ptr));  // 转移到 deAllocator_ 指向
  }
  dataframe->ctx = DevContext(decode_frame->dev_type, decode_frame->device_id);
  dataframe->fmt = CNDataFormat::CN_PIXEL_FORMAT_BGR24;  // 统一转换为 BGR24 格式
  // 2. 创建平台相关 memop 来拷贝图像内存
  dataframe->CopyToSyncMem(decode_frame);
  return 0;
}