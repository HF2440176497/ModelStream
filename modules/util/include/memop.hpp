
#ifndef MEMOP_HPP_
#define MEMOP_HPP_

#include <memory>
#include <map>
#include <mutex>

#include "cnstream_syncmem.hpp"

namespace cnstream {

/**
 * @struct DevContext
 *
 * @brief DevContext is a structure holding the information that CNDataFrame data is allocated by CPU or MLU.
 */
struct DevContext {
  DevContext() = default;
  DevContext(DevType type, int id) : dev_type(type), dev_id(id) {}
  DevType dev_type = DevType::INVALID; /*!< Device type. The default value is ``INVALID``.*/
  int dev_id = -1;                /*!< Ordinal device ID. */
};


struct Buffer {
  std::shared_ptr<void> data;
  size_t size;
  int device_id;
  Buffer(std::shared_ptr<void> d, size_t s, int dev) 
      : data(std::move(d)), size(s), device_id(dev) {}
};

/**
 * 用于包含多平台分配的内存
 */
class MemoryBufferCollection {
 public:
  Buffer& GetBuffer(DevType type, size_t size, int device_id = -1);
  bool Has(DevType type) const;
  Buffer* Get(DevType type);
  void Clear(DevType type);
  void ClearAll();

 private:
  std::map<DevType, Buffer> buffers_;
  std::mutex mutex_;
};

/**
 * @brief 向上取为 64 KB 的整数倍
 */
static inline size_t RoundUpSize(size_t bytes) {
  const size_t alignment = 64 * 1024;
  return (bytes + alignment - 1) / alignment * alignment;
}

// 前向声明，避免循环依赖
class DecodeFrame;


/**
 * @brief 内存操作算子
 * MemOp 是用于描述在某设备上的
 * 
 */
class MemOp {
 public:
  virtual std::shared_ptr<void> Allocate(size_t bytes);  // 分配操作算子的目标内存，内含 RAII 管理
  virtual void Copy(void* dst, const void* src, size_t size);
  virtual int GetDeviceId() const;
  virtual std::shared_ptr<CNSyncedMemory> CreateSyncedMemory(size_t size);
  virtual void SetData(std::shared_ptr<CNSyncedMemory> mem, void* data);  // 将分配的目标内存绑定到 CNSyncedMemory
  virtual int ConvertImageFormat(void* dst, CNDataFormat dst_fmt, const DecodeFrame* src_frame);
};

static bool RegisterMemOp() {
  auto& factory = MemOpFactory::Instance();
  bool result = true;
  result &= factory.RegisterMemOpCreator(DevType::CPU, 
    [](int dev_id) {
      return std::make_unique<MemOp>();
    });
  return result;
}

static bool memop_registered = RegisterMemOp();

}  // namespace cnstream

#endif