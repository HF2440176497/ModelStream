
#include "frame_server.h"
#include "decode.h"
#include "utils.h"
#include "decode_server.h"


void basic_func() {
    int ret;
    std::cout << "main process start" << std::endl;

    StreamInfo stream_info = StreamInfo();
    stream_info.protocol = RTMP;
    stream_info.url = "rtmp://localhost/live/stream_test";

    std::shared_ptr<Decode> p = nullptr;
    std::vector<std::shared_ptr<Decode>> list;

    for (int i = 0; i < 4; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "--------------------- current time: " << i << " ---------------------" << std::endl;

        /**
         * 创建一个线程执行
         */
        p = std::make_shared<Decode>(stream_info);
        list.push_back(p);
        std::thread external_thread(&Decode::get_frame, p.get());
        external_thread.detach();

        if ((ret = p->init()) != 0) {
            std::cout << "current time: " << i << "; connect fail" << std::endl;
            p->release();
            continue;
        }
        for (int j = 0; j < 20; ++j) {
            Frame res = p->read();
            // std::cout << "main: read return" << std::endl;
            if (res.ret == SUCCESS) {
                auto img = res.frame;
                std::cout << "main get image height: " << img.rows << "; image width: " << img.cols << std::endl;
            }           
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

void server_func() {
    StreamInfo stream_info = StreamInfo();
    stream_info.protocol = RTMP;
    stream_info.url = "rtmp://localhost/live/stream_test";

    DecodeServer server(stream_info);
    if (server.init() != 0) 
        return;
    server.test();
}

int main() {
    server_func();
    return 0;
}

