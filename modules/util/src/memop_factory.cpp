
#include "memop_factory.hpp"
#include "util/cnstream_logging.hpp"

namespace cnstream {

MemOpFactory& MemOpFactory::Instance() {
  static MemOpFactory instance;
  return instance;
}

bool MemOpFactory::RegisterMemOpCreator(DevType dev_type,
                                        std::function<std::unique_ptr<MemOp>(int dev_id)> creator) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = creators_.insert({dev_type, creator});
  return inserted;
}

std::unique_ptr<MemOp> MemOpFactory::CreateMemOp(DevType dev_type, int dev_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = creators_.find(dev_type);
  if (it != creators_.end()) {
    return it->second(dev_id);
  }
  return nullptr;
}

}  // namespace cnstream