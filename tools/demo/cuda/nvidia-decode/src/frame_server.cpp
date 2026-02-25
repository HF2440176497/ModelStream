
#include "utils.h"
#include "frame_server.h"


FrameServer::FrameServer() { }

/**
 * 析构函数：释放含有的引用计数 避免内存泄漏
 */
FrameServer::~FrameServer() {
    while (!frame_queue.empty()) {
        frame_queue.pop();
    }
}


Frame FrameServer::get() {
    Frame res = Frame();
    _queue_mtx.lock();
    if (frame_queue.size() == 0) {
        res.ret = NONE;
        std::cout << "get frame_queue.size == 0" << std::endl;
    } else {
        res = frame_queue.front();
        frame_queue.pop();
    }
    _queue_mtx.unlock();
    return res;
}

/**
 * 非阻塞向队列放入 Frame 对象
 */
void FrameServer::put(Frame frame) {
    _queue_mtx.lock();
    if (frame_queue.size() >= MAX_QUEUE_SIZE) {
        frame_queue.pop();
    }
    frame_queue.push(frame);  // 对象拷贝，导致智能指针拷贝，保证生命周期
    std::cout << "put frame_queue.size: " << frame_queue.size() << std::endl;
    _queue_mtx.unlock();
}
