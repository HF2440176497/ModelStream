#include "utils.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

int64_t get_timestamp_ms() {
    auto now = std::chrono::high_resolution_clock::now();  // 获取当前时刻的高精度时间点
    auto duration_since_epoch = now.time_since_epoch();    // 转换为自纪元（1970-01-01 00:00:00 UTC）以来的毫秒数
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration_since_epoch);
    return milliseconds.count();  // 返回毫秒数
}

int64_t get_timestamp_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration_since_epoch = now.time_since_epoch();
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration_since_epoch);
    return microseconds.count();  // 返回微秒数
}

/**
 * 将微秒转换为自然时间格式化打印
 */
void print_actual_time(int64_t microseconds, std::string desc_str) {
    auto time_point = std::chrono::system_clock::time_point(std::chrono::microseconds(microseconds));
    std::time_t time = std::chrono::system_clock::to_time_t(time_point);
    std::tm tm = *std::localtime(&time);
    std::cout << desc_str << "Natural time: " << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << std::endl;
}
