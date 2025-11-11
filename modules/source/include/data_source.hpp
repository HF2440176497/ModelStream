

#ifndef MODULES_DATA_SOURCE_HPP_
#define MODULES_DATA_SOURCE_HPP_


#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cnstream_frame.hpp"
#include "cnstream_frame_va.hpp"
#include "cnstream_pipeline.hpp"
#include "cnstream_source.hpp"
#include "data_source_param.hpp"


namespace cnstream {

/*!
 * @class DataSource
 *
 * @brief DataSource is a class to handle encoded input data.
 *
 * @note It is always the first module in a pipeline.
 */
class DataSource : public SourceModule, public ModuleCreator<DataSource> {
 public:
  /*!
   * @brief Constructs a DataSource object.
   *
   * @param[in] moduleName The name of this module.
   *
   * @return No return value.
   */
  explicit DataSource(const std::string &moduleName);

  /*!
   * @brief Destructs a DataSource object.
   *
   * @return No return value.
   */
  ~DataSource();

  /*!
   * @brief Initializes the configuration of the DataSource module.
   *
   * This function will be called by the pipeline when the pipeline starts.
   *
   * @param[in] paramSet The module's parameter set to configure a DataSource module.
   *
   * @return Returns true if the parammeter set is supported and valid, othersize returns false.
   */
  bool Open(ModuleParamSet paramSet) override;
  // override Module's virtual function

  /*!
   * @brief Frees the resources that the object may have acquired.
   *
   * This function will be called by the pipeline when the pipeline stops.
   *
   * @return No return value.
   */
  void Close() override;

  /*!
   * @brief Checks the parameter set for the DataSource module.
   *
   * @param[in] paramSet Parameters for this module.
   * 
   * @return Returns true if all parameters are valid. Otherwise, returns false.
   * 
   * @note DataSource::Open 调用
   */
  bool CheckParamSet(const ModuleParamSet &paramSet) const override;

  /*!
   * @brief Gets the parameters of the DataSource module.
   *
   * @return Returns the parameters of this module.
   *
   * @note This function should be called after ``Open`` function.
   */
  DataSourceParam GetSourceParam() const { return param_; }

 private:
  DataSourceParam param_;
};  // class DataSource

REGISTER_MODULE(DataSource);  // 启动反射

// 派生关系: Module SourceModule DataSource
// SourceModule 并没有提供虚函数接口, DataSource 主要重写 Module 的相关 virtual func

class ImageQueueHandlerImpl;

/**
 * 图像队列接口
 */
class ImageQueueHandler : public SourceHandler {

 public:
  static std::shared_ptr<SourceHandler> Create(DataSource *module, const std::string &stream_id);

  /**
   * 通过 SourceModule::AddSource 调用
   */
  bool Open() override;  // pure virtual function
  void Stop() override;
  void Close() override;

 public:
  void PushDatas(std::vector<uint64_t> timestamps, std::vector<cv::Mat> images);

 private:
  explicit ImageQueueHandler(DataSource *module, const std::string &stream_id): SourceHandler(module, stream_id);

#ifdef UNIT_TEST
 public:
#endif
  std::unique_ptr<ImageQueueHandlerImpl> impl_;
};

}  // namespace cnstream

#endif