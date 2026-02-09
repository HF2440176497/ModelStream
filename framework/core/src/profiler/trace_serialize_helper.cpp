/*************************************************************************
 * Copyright (C) [2019] by Cambricon, Inc. All rights reserved
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

#include <string>
#include <utility>

#include "profiler/trace_serialize_helper.hpp"

namespace cnstream {

bool TraceSerializeHelper::DeserializeFromJSONStr(const std::string& jsonstr, TraceSerializeHelper* pout) {
  try {
    nlohmann::json doc = nlohmann::json::parse(jsonstr);
    if (!doc.is_array()) return false;
    pout->doc_ = std::move(doc);
  } catch (const nlohmann::json::parse_error& e) {
    LOGE(PROFILER) << "Parse trace data failed. Error: " << e.what();
    return false;
  }
  return true;
}

bool TraceSerializeHelper::DeserializeFromJSONFile(const std::string& filename, TraceSerializeHelper* pout) {
  std::ifstream ifs(filename);
  if (!ifs.is_open()) {
    LOGE(PROFILER) << "File open failed :" << filename;
    return false;
  }
  std::string jstr((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
  ifs.close();
  if (!TraceSerializeHelper::DeserializeFromJSONStr(jstr, pout)) {
    return false;
  }
  return true;
}

TraceSerializeHelper::TraceSerializeHelper() {
  doc_ = nlohmann::json::array();
}

TraceSerializeHelper::TraceSerializeHelper(const TraceSerializeHelper& t) {
  *this = t;
}

TraceSerializeHelper::TraceSerializeHelper(TraceSerializeHelper&& t) {
  *this = std::forward<TraceSerializeHelper>(t);
}

TraceSerializeHelper& TraceSerializeHelper::operator=(const TraceSerializeHelper& t) {
  doc_ = t.doc_;
  return *this;
}

TraceSerializeHelper& TraceSerializeHelper::operator=(TraceSerializeHelper&& t) {
  doc_ = std::move(t.doc_);
  return *this;
}

/**
 * 调用处：TraceSerializeHelper::Serialize
 */
static
nlohmann::json GenerateProcessProfileValue(const ProcessProfile& profile, const std::string& module_name) {
  nlohmann::json value;
  value["module_name"] = module_name;
  value["process_name"] = profile.process_name;
  value["counter"] = profile.counter;
  value["completed"] = profile.completed;
  value["dropped"] = profile.dropped;
  value["avg_latency"] = profile.avg_latency;
  value["max_latency"] = profile.max_latency;
  value["min_latency"] = profile.min_latency;
  value["fps"] = profile.fps;
  return value;
}

void TraceSerializeHelper::Serialize(const ModuleProfile& module_profile) {
  nlohmann::json module_info;
  module_info["module_name"] = module_profile.module_name;
  
  nlohmann::json processes = nlohmann::json::array();
  for (const auto& process_profile : module_profile.process_profiles) {
    processes.push_back(GenerateProcessProfileValue(process_profile, module_profile.module_name));
  }
  module_info["process_profiles"] = processes;
  doc_.push_back(module_info);
}

void TraceSerializeHelper::Merge(const TraceSerializeHelper& t) {
  for (const auto& value : t.doc_) {
    doc_.push_back(value);
  }
}

std::string TraceSerializeHelper::ToJsonStr() const {
  return doc_.dump();
}

void TraceSerializeHelper::Reset() {
  *this = TraceSerializeHelper();
}

}  // namespace cnstream