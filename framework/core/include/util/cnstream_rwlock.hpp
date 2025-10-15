
#ifndef CNSTREAM_RWLOCK_H_
#define CNSTREAM_RWLOCK_H_

#include <thread>
#include <memory>
#include <shared_mutex>

namespace cnstream {

class RwLock {
 public:
  RwLock() = default;
  ~RwLock() = default;
  
  void wrlock() { mutex_.lock(); }
  void rdlock() { mutex_.lock_shared(); }
  
  void wrunlock() { mutex_.unlock(); }        // 写锁解锁
  void rdunlock() { mutex_.unlock_shared(); } // 读锁解锁

 private:
  std::shared_mutex mutex_;
};

class RwLockWriteGuard {
 public:
  explicit RwLockWriteGuard(RwLock& lock) : lock_(lock) { lock_.wrlock(); }
  ~RwLockWriteGuard() { lock_.wrunlock(); }

 private:
  RwLock& lock_;
};

class RwLockReadGuard {
 public:
  explicit RwLockReadGuard(RwLock& lock) : lock_(lock) { lock_.rdlock(); }
  ~RwLockReadGuard() { lock_.rdunlock(); }

 private:
  RwLock& lock_;
};

} /* namespace cnstream */

#endif /* CNSTREAM_RWLOCK_H_ */
