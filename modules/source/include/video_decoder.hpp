/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *************************************************************************/

#ifndef CNSTREAM_VIDEO_DECODER_HPP_
#define CNSTREAM_VIDEO_DECODER_HPP_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "util/memop.hpp"
#include "video_parser.hpp"
#include "video_decoder.hpp"

namespace cnstream {

static constexpr int MAX_PLANE_NUM = 3;

class IDecBufRef {
public:
  virtual ~IDecBufRef() = default;
};

/**
 * @brief 零拷贝内存引用器 作为 DecodeFrame 的 buf_ref
 */
class Deallocator : public IDataDeallocator {
 public:
  explicit Deallocator(IDecBufRef *ptr) {
    ptr_.reset(ptr);
  }
  ~Deallocator() {
    ptr_.reset();
  }
 private:
  std::unique_ptr<IDecBufRef> ptr_;
};

struct DecodeFrame {
  DecodeFrame(int height, int width, PixFmt fmt) : height(height), width(width), fmt(fmt) {
    valid = true;
    pts = 0;
  }
  bool valid;
  int64_t pts;
  //  the below parameters work when 'valid' set true.
  int32_t height;
  int32_t width; 
  CNDataFormat fmt;
  DevType dev_type = DevType::INVALID;  // 表示该帧数据位置
  int32_t device_id;
  int32_t planeNum;
  void *plane[MAX_PLANE_NUM];
  int stride[MAX_PLANE_NUM];
  std::unique_ptr<IDecBufRef> buf_ref = nullptr;

 public:
  ~DecodeFrame() {}
};

struct ExtraDecoderInfo {
  // for mlu decoders
  int32_t device_id = 0;
  int32_t input_buf_num = 2;  // for MLU200
  int32_t output_buf_num = 4;
  bool apply_stride_align_for_scaler = false;  // for MLU220
  int32_t max_width = 0;   // for jpu
  int32_t max_height = 0;  // for jpu
  std::vector<uint8_t> extra_info;
};

class IDecodeResult {
 public:
  virtual ~IDecodeResult() = default;
  virtual void OnDecodeError(DecodeErrorCode error_code) {}
  virtual void OnDecodeFrame(DecodeFrame *frame) = 0;
  virtual void OnDecodeEos() = 0;
};

class Decoder {
 public:
  explicit Decoder(const std::string& stream_id, IDecodeResult *cb) : stream_id_(stream_id), result_(cb) {}
  virtual ~Decoder() = default;
  virtual bool Create(VideoInfo *info, ExtraDecoderInfo *extra = nullptr) = 0;
  virtual bool Process(VideoEsPacket *pkt)  = 0;
  virtual void Destroy() = 0;

 protected:
  std::string stream_id_ = "";
  IDecodeResult *result_;
};

}  // namespace cnstream

#endif  // CNSTREAM_VIDEO_DECODER_HPP_
