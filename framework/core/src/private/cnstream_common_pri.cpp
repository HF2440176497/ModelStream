
#include "cnstream_logging.hpp"
#include "private/cnstream_common_pri.hpp"

namespace cnstream {

// ----------- Some related function definitions; Originally located at framework/src/cntream_frame.cpp

/**
 * sync == true: 找到 steam_id 的情况下一直等到 second == true 才返回
 */
bool CheckStreamEosReached(const std::string &stream_id, bool sync) {
  if (sync) {
    while (1) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      std::lock_guard<std::mutex> guard(s_eos_lock_);
      auto iter = s_stream_eos_map_.find(stream_id);
      if (iter != s_stream_eos_map_.end()) {
        if (iter->second == true) {
          s_stream_eos_map_.erase(iter);
          LOGI(CORE) << "check stream eos reached, stream_id =  " << stream_id;
          return true;
        }
      } else {
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

void SetStreamRemoved(const std::string &stream_id, bool value) {
  std::lock_guard<std::mutex> guard(s_remove_lock_);
  auto iter = s_stream_removed_map_.find(stream_id);
  if (iter != s_stream_removed_map_.end()) {
    if (value != true) {
      s_stream_removed_map_.erase(iter);
      return;
    }
    iter->second = true;
  } else {
    s_stream_removed_map_[stream_id] = value;
  }
  LOGI(CORE) << "_____SetStreamRemoved " << stream_id << ":" << s_stream_removed_map_[stream_id];
}


bool IsStreamRemoved(const std::string &stream_id) {
  std::lock_guard<std::mutex> guard(s_remove_lock_);
  auto iter = s_stream_removed_map_.find(stream_id);
  if (iter != s_stream_removed_map_.end()) {
    LOGI(CORE) << "_____IsStreamRemoved " << stream_id << ":" << s_stream_removed_map_[stream_id];
    return s_stream_removed_map_[stream_id];
  }
  return false;
}

}  // namespace cnstream