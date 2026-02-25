
// 图像帧管理器
#ifndef _FRAMESERVER_
#define _FRAMESERVER_

#include <queue>
#include <mutex>
#include <memory>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include "utils.h"

#define SUCCESS   0
#define NONE     -1
#define ERROR    -2

#define MAX_QUEUE_SIZE 50

class Frame {
public:
    int ret;
    cv::Mat frame;
    int64_t absolute_timestamp_us;
    int64_t absolute_timestamp_ms;
};


class FrameServer {

public:
    FrameServer();
    ~FrameServer();

public:
    void put(Frame frame);
    Frame get();

private:
    std::queue<Frame> frame_queue;
    std::mutex _queue_mtx;
};

#endif