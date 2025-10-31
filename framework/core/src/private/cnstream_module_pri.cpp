
#include "cnstream_logging.hpp"
#include "private/cnstream_module_pri.hpp"

namespace cnstream {

uint32_t GetMaxStreamNumber() { return MAX_STREAM_NUM; }

uint32_t GetMaxModuleNumber() {
  /*maxModuleIdNum is sizeof(module_id_mask_) * 8  (bytes->bits)*/
  return sizeof(uint64_t) * 8;
}

uint32_t IdxManager::GetStreamIndex(const std::string& stream_id) {
  std::lock_guard<std::mutex> guard(id_lock);
  auto search = stream_idx_map.find(stream_id);
  if (search != stream_idx_map.end()) {
    return search->second;
  }

  for (uint32_t i = 0; i < GetMaxStreamNumber(); i++) {
    if (!stream_bitset[i]) {
      stream_bitset.set(i);
      stream_idx_map[stream_id] = i;
      return i;
    }
  }
  return INVALID_STREAM_IDX;
}

void IdxManager::ReturnStreamIndex(const std::string& stream_id) {
  std::lock_guard<std::mutex> guard(id_lock);
  auto search = stream_idx_map.find(stream_id);
  if (search == stream_idx_map.end()) {
    return;
  }
  uint32_t stream_idx = search->second;
  if (stream_idx >= GetMaxStreamNumber()) {
    return;
  }
  stream_bitset.reset(stream_idx);
  stream_idx_map.erase(search);
}

size_t IdxManager::GetModuleIdx() {
  std::lock_guard<std::mutex>  guard(id_lock);
  for (size_t i = 0; i < GetMaxModuleNumber(); i++) {
    if (!(module_id_mask_ & ((uint64_t)1 << i))) {
      module_id_mask_ |= (uint64_t)1 << i;
      return i;
    }
  }
  return INVALID_MODULE_ID;
}

void IdxManager::ReturnModuleIdx(size_t id_) {
  std::lock_guard<std::mutex>  guard(id_lock);
  if (id_ >= GetMaxModuleNumber()) {
    return;
  }
  module_id_mask_ &= ~(1 << id_);
}

}  // namespace cnstream