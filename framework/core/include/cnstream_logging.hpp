/*************************************************************************
* Copyright (C) [2019-2022] by Cambricon, Inc. All rights reserved
*
* This source code is licensed under the Apache-2.0 license found in the
* LICENSE file in the root directory of this source tree.
*
* A part of this source code is referenced from glog project.
* https://github.com/google/glog/blob/master/src/logging.cc
*
* Copyright (c) 1999, Google Inc.
*
* This source code is licensed under the BSD 3-Clause license found in the
* LICENSE file in the root directory of this source tree.
*
*************************************************************************/

#ifndef CNSTREAM_LOGGING_HPP_
#define CNSTREAM_LOGGING_HPP_
#include <glog/logging.h>

#define LOGF(tag) LOG(FATAL) << "[" << (#tag) << " FATAL] "
#define LOGE(tag) LOG(ERROR) << "[" << (#tag) << " ERROR] "
#define LOGW(tag) LOG(WARNING) << "[" << (#tag) << " WARN] "
#define LOGI(tag) LOG(INFO) << "[" << (#tag) << " INFO] "
#define LOGD(tag) VLOG(1) << "[" << (#tag) << " DEBUG] "
#define LOGT(tag) VLOG(2) << "[" << (#tag) << " TRACE] "

#define LOGF_IF(tag, condition) LOG_IF(FATAL, condition) << "[" << (#tag) << " FATAL] "
#define LOGE_IF(tag, condition) LOG_IF(ERROR, condition) << "[" << (#tag) << " ERROR] "
#define LOGW_IF(tag, condition) LOG_IF(WARNING, condition) << "[" << (#tag) << " WARN] "
#define LOGI_IF(tag, condition) LOG_IF(INFO, condition) << "[" << (#tag) << " INFO] "
#define LOGD_IF(tag, condition) VLOG_IF(1, condition) << "[" << (#tag) << " DEBUG] "
#define LOGT_IF(tag, condition) VLOG_IF(2, condition) << "[" << (#tag) << " TRACE] "

#endif  // CNSTREAM_LOGGING_HPP_
