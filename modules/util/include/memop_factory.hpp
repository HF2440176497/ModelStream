#ifndef MEMOP_FACTORY_HPP_
#define MEMOP_FACTORY_HPP_

#include <memory>
#include <unordered_map>
#include <functional>
#include <mutex>

#include "memop.hpp"

namespace cnstream {

/**
 * @class MemOpFactory
 * @brief 用于创建MemOp实例的工厂类，支持不同硬件平台的MemOp扩展
 * 
 * MemOpFactory采用单例模式，允许不同硬件平台的MemOp实现通过注册机制加入，而不需要修改通用代码。
 */
class MemOpFactory {
public:
  /**
   * @brief 获取MemOpFactory的单例实例
   * @return 返回MemOpFactory的唯一实例
   */
  static MemOpFactory& Instance();

  /**
   * @brief 注册MemOp创建函数
   * @param src_dev_type 源设备类型
   * @param dst_dev_type 目标设备类型
   * @param creator 创建MemOp实例的函数
   * @return 注册是否成功
   */
  bool RegisterMemOpCreator(DevType dev_type,
                           std::function<std::unique_ptr<MemOp>(int dev_id)> creator);

  /**
   * @brief 根据设备类型创建MemOp实例
   * @param dev_type 设备类型
   * @param dev_id 设备ID
   * @return 返回创建的MemOp实例，如果不支持该设备类型则返回nullptr
   */
  std::unique_ptr<MemOp> CreateMemOp(DevType dev_type, int dev_id = -1);

private:
  /**
   * @brief 私有构造函数，防止外部创建实例
   */
  MemOpFactory() = default;

  /**
   * @brief 析构函数
   */
  ~MemOpFactory() = default;

  /**
   * @brief 禁止拷贝构造函数
   */
  MemOpFactory(const MemOpFactory&) = delete;

  /**
   * @brief 禁止赋值操作符
   */
  MemOpFactory& operator=(const MemOpFactory&) = delete;

#ifdef UNIT_TEST
  void PrintRegisteredCreators() {
    LOGI(MEMOP_FACTORY) << "PrintRegisteredCreators size: " << creators_.size();
    for (const auto& pair : creators_) {
      LOGI(MEMOP_FACTORY) << "DevType: " << static_cast<int>(pair.first)
                      << " -> Creator Function Address: " << &pair.second;
    }
  }
#endif

private:
  struct DevTypeHash {
    template <typename T>
    std::size_t operator()(const T& dev_type) const {
      return static_cast<std::size_t>(dev_type);
    }
  };

  std::unordered_map<DevType, std::function<std::unique_ptr<MemOp>(int dev_id)>, DevTypeHash> creators_;
  std::mutex mutex_;
};

}  // namespace cnstream

#endif  // MEMOP_FACTORY_HPP_