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

#ifndef CNSTREAM_COMMON_PRI_HPP_
#define CNSTREAM_COMMON_PRI_HPP_

#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <map>
#include <atomic>
#include <filesystem>
#include <set>

#include "private/cnstream_constants_pri.hpp"


namespace cnstream {

/*!
 * @class NonCopyable
 *
 * @brief NonCopyable is the abstraction of the class which has no ability to do copy and assign. It is always be used
 *        as the base class to disable copy and assignment.
 */
class NonCopyable {
 protected:
  NonCopyable() = default;
  ~NonCopyable() = default;

 private:
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable(NonCopyable&&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
  NonCopyable& operator=(NonCopyable&&) = delete;
};  // end NonCopyable


// some static variables for stream EOS and removed status
inline std::mutex s_eos_lock_;
inline std::map<std::string, std::atomic<bool>> s_stream_eos_map_;

inline std::mutex s_remove_lock_;
// static std::map<std::string, bool> s_stream_removed_map_;
inline std::set<std::string> s_stream_removed_set_;


/**
 * @brief Checks one stream whether reaches EOS.
 *
 * @param[in] stream_id The identifier of a stream.
 * @param[in] sync The mode of checking the status. True means checking in synchronized mode while False represents
 *                 for asynchronous.
 *
 * @return Returns true if the EOS reached, otherwise returns false.
 *
 * @note It's used for removing sources forcedly.
 */
bool CheckStreamEosReached(const std::string &stream_id, bool sync = true);
/**
 * @brief Checks one stream whether reaches EOS.
 *
 * @param[in] stream_id The identifier of a stream.
 * @param[in] value The status of a stream.
 *
 * @return No return value.
 *
 * @note It's used for removing sources forcedly.
 */
void SetStreamRemoved(const std::string &stream_id, bool value = true);
/**
 * @brief Checks whether a stream is removed.
 *
 * @param[in] stream_id The identifier of a stream.
 *
 * @return Returns true if the stream is removed, otherwise returns false.
 *
 * @note It's used for removing sources forcedly.
 */
bool IsStreamRemoved(const std::string &stream_id);

void PrintStreamEos();

bool StreamEosMapValue(const std::string &stream_id);

/**
 * @brief
 */
inline std::vector<std::string> StringSplit(const std::string& s, char c) {
  std::stringstream ss(s);
  std::string piece;
  std::vector<std::string> result;
  while (std::getline(ss, piece, c)) {
    result.push_back(piece);
  }
  return result;
}

/**
 * @brief Creates directories if not exists.
 *
 * @param[in] path The path of directories.
 *
 * @return Returns true if the directories have been created successfully, otherwise returns false.
 */
inline bool create_directories_if_not_exists(const std::string& path) {
    if (std::filesystem::exists(path)) { return true; }
    try {
        return std::filesystem::create_directories(path);
    } catch (const std::filesystem::filesystem_error& e) {
        return false;
    } catch (const std::exception& e) {
        return false;
    }
    return false;
}

}  // namespace cnstream
#endif  // CNSTREAM_COMMON_PRI_HPP_
