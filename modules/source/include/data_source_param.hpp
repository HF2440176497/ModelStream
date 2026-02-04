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

/*!
 * @enum OutputType
 * @brief Enumeration variables describing the storage type of the output frame data of a module.
 */
enum class OutputType {
  OUTPUT_CPU,  /*!< CPU is the used storage type. */
  OUTPUT_MLU   /*!< MLU is the used storage type. */
};

/*!
 * @enum DecoderType
 * @brief Enumeration variables describing the decoder type used in source module.
 */
enum class DecoderType {
  DECODER_CPU,  /*!< CPU decoder is used. */
  DECODER_MLU   /*!< MLU decoder is used. */
};

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
