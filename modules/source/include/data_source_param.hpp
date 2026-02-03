/*************************************************************************
 * Copyright (C) [2021] by Cambricon, Inc. All rights reserved
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

#ifndef MODULES_DATA_SOURCE_PARAM_HPP_
#define MODULES_DATA_SOURCE_PARAM_HPP_

namespace cnstream {

enum class DevType {
  INVALID = -1,                /*!< Invalid device type. */
  CPU = 0,                     /*!< The data is allocated by CPU. */
  CUDA = 1,                    /*!< The data is allocated by CUDA. */
  NPU = 2,                     /*!< The data is allocated by NPU. */
};

inline constexpr std::unordered_map<std::string, DevType> StrDevTypeMap = {
  {"CPU", DevType::CPU},
  {"CUDA", DevType::CUDA},
  {"NPU", DevType::NPU},
};

/*!
 * @enum DecoderType
 * @brief Enumeration variables describing the decoder type used in source module.
 */
enum class DecoderType {
  DECODER_CPU,  /*!< CPU decoder is used. */
  DECODER_CUDA, /*!< CUDA decoder is used. */
  DECODER_NPU,  /*!< NPU decoder is used. */
};

inline constexpr std::unordered_map<std::string, DecoderType> StrDecoderTypeMap = {
  {"CPU", DecoderType::DECODER_CPU},
  {"CUDA", DecoderType::DECODER_CUDA},
  {"NPU", DecoderType::DECODER_NPU},
};

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
  CN_PIXEL_FORMAT_BGRA32,          /*!< This frame is in the BGRA32 format. */
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

/*!
 * @brief DataSourceParam is a structure for private usage.
 */
struct DataSourceParam {
  int  device_id_ = -1;                                 /*! DataFrame 的 dev_id 直接来自 decode_frame  */
  CNDataFormat output_type_ = CNDataFormat::CN_PIXEL_FORMAT_BGR24;  /*!< The output type */
  size_t  interval_ = 1;                                /*!< The interval of outputting one frame. It outputs one frame every n (interval_) frames. */
  DecoderType decoder_type_ = DecoderType::DECODER_CPU; /*!< The decoder type. */
  bool only_key_frame_ = false;                         /*!< Whether only to decode key frames. */
  std::string file_path_ = "";                          /*!< The file path of the video or image file. */
};
}  // namespace cnstream

#endif  // MODULES_DATA_SOURCE_PARAM_HPP_
