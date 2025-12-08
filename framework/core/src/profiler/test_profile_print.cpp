/*************************************************************************
 * Copyright (C) [2020] by Cambricon, Inc. All rights reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

#include "profiler/profile.hpp"
#include <iostream>

int main() {
  // Create a ProcessProfile for testing
  cnstream::ProcessProfile process_profile;
  process_profile.process_name = "inference_process";
  process_profile.counter = 100;
  process_profile.completed = 95;
  process_profile.dropped = 5;
  process_profile.avg_latency = 12.5;
  process_profile.max_latency = 25.0;
  process_profile.min_latency = 5.0;
  process_profile.fps = 80.0;
  
  // Create a ModuleProfile for testing
  cnstream::ModuleProfile module_profile;
  module_profile.module_name = "inference_module";
  module_profile.counter = 200;
  module_profile.completed = 185;
  module_profile.dropped = 15;
  module_profile.process_profiles.push_back(process_profile);
  
  // Test ProcessProfileToString function
  std::cout << "=== Testing ProcessProfileToString ===\n";
  std::cout << cnstream::ProcessProfileToString(process_profile) << std::endl;
  
  // Test ModuleProfileToString function
  std::cout << "=== Testing ModuleProfileToString ===\n";
  std::cout << cnstream::ModuleProfileToString(module_profile) << std::endl;
  
  // Test operator<< for ProcessProfile
  std::cout << "=== Testing operator<< for ProcessProfile ===\n";
  std::cout << process_profile << std::endl;
  
  // Test operator<< for ModuleProfile
  std::cout << "=== Testing operator<< for ModuleProfile ===\n";
  std::cout << module_profile << std::endl;
  
  return 0;
}
