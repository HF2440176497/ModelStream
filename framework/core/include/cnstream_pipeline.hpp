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

#ifndef CNSTREAM_PIPELINE_HPP_
#define CNSTREAM_PIPELINE_HPP_

/**
 * @file cnstream_pipeline.hpp
 *
 * This file contains a declaration of the Pipeline class.
 */

#include <atomic>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <map>
#include <utility>
#include <vector>


#include "cnstream_common.hpp"
#include "cnstream_config.hpp"
#include "cnstream_eventbus.hpp"
#include "cnstream_graph.hpp"


namespace cnstream {
    

struct NodeContext {
  // std::shared_ptr<Module> module;
  // std::shared_ptr<Connector> connector;
  uint64_t parent_nodes_mask = 0;
  uint64_t route_mask = 0;  // for head nodes
  // for gets node instance by a module, see Module::context_;
  std::weak_ptr<CNGraph<NodeContext>::CNNode> node;
};


}

#endif  // CNSTREAM_PIPELINE_HPP_