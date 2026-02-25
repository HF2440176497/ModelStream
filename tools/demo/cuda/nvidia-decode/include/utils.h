#ifndef _UTILS_H_
#define _UTILS_H_

#include <chrono>
#include <iostream>

class StreamInfo {
public:
    StreamInfo() {};
    ~StreamInfo() {};

public:
    int protocol;
    std::string url;
};

int64_t get_timestamp_ms();
int64_t get_timestamp_us();

void print_actual_time(int64_t microseconds, std::string desc_str);

#endif
