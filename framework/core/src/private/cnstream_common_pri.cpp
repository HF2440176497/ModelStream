
#include <iostream>

#include "cnstream_logging.hpp"
#include "private/cnstream_common_pri.hpp"

namespace cnstream {

// ----------- Some related function definitions; Originally located at framework/src/cntream_frame.cpp

/**
 * sync == true: 找到 steam_id 的情况下一直等到 second == true 才返回
 * s_stream_eos_map_ 仅在 FrameInfo 析构函数中设置 
 * @return true: 查询到 second == true 移除成功
 */
bool CheckStreamEosReached(const std::string &stream_id, bool sync) {
  if (sync) {
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::lock_guard<std::mutex> guard(s_eos_lock_);
      auto iter = s_stream_eos_map_.find(stream_id);
      if (iter != s_stream_eos_map_.end()) {
        if (iter->second == true) {
          s_stream_eos_map_.erase(iter);
          LOGI(CORE) << "check stream eos reached, stream_id = " << stream_id;
          return true;
        }
      } else {
        LOGW(CORE) << "check stream eos, stream_eos_map not found " << stream_id;
        return false;
      }
    }
    return false;
  } else {
    std::lock_guard<std::mutex> guard(s_eos_lock_);
    auto iter = s_stream_eos_map_.find(stream_id);
    if (iter != s_stream_eos_map_.end()) {
      if (iter->second == true) {
        s_stream_eos_map_.erase(iter);
        return true;
      }
    }
    return false;
  }
}

// void SetStreamRemoved(const std::string &stream_id, bool value) {
//   std::lock_guard<std::mutex> guard(s_remove_lock_);
//   auto iter = s_stream_removed_map_.find(stream_id);
//   if (iter != s_stream_removed_map_.end()) {
//     if (value != true) {
//       s_stream_removed_map_.erase(iter);
//       return;
//     }
//     iter->second = true;
//   } else {
//     s_stream_removed_map_[stream_id] = value;
//   }
//   LOGI(CORE) << "_____SetStreamRemoved " << stream_id << ":" << s_stream_removed_map_[stream_id];
// }


/**
 * value == true: 设置移除标志
 * value == false: 删除移除标志
 * 当我们确定不再需要的时候的，才可以放心的删除 stream_id；RemoveSource 调用
 */
void SetStreamRemoved(const std::string &stream_id, bool value) {
  std::lock_guard<std::mutex> guard(s_remove_lock_);
  if (value) {  // 设置移除标志
    s_stream_removed_set_.insert(stream_id);
  } else {
    auto iter = s_stream_removed_set_.find(stream_id);
    if (iter != s_stream_removed_set_.end()) {
      s_stream_removed_set_.erase(stream_id);
    }
  }
}

/**
 * 是否是正在移除的 streamid
 */
bool IsStreamRemoved(const std::string &stream_id) {
  std::lock_guard<std::mutex> guard(s_remove_lock_);
  bool removed = s_stream_removed_set_.find(stream_id) != s_stream_removed_set_.end();
  return removed;
}

void PrintStreamEos() { 
  std::cout << "PrintStreamEos: ";
  std::lock_guard<std::mutex> guard(s_eos_lock_);
  for (const auto& pair : s_stream_eos_map_) {
      std::cout << pair.first << ": " << std::boolalpha << pair.second.load() << ", ";
  }
  std::cout << std::endl;
}

}  // namespace cnstream
