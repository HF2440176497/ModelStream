
/**
 * 编解码线程池
 * 将 FrameServer 注入到 Decode 中，只含有一个 frame_server 
 * 接收
 */

#include "decode_server.h"


DecodeServer::DecodeServer(StreamInfo in_stream) : in_stream(in_stream) {
}

DecodeServer::~DecodeServer() {
    for (auto item: decoders) {
        item->release();
    }
}



/**
 * 扩展：添加监控线程，对于非活跃 decode 对象，检查 is_finish 标志位，从 pool 中清除
 */

int DecodeServer::init() {
    for (int i = 0; i < max_reconnect_num; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "--------------------- current time: " << i << " ---------------------" << std::endl;
        auto p = std::make_shared<Decode>(in_stream);
        decoders.push_back(p);

        std::thread external_thread(&Decode::get_frame, p.get());
        external_thread.detach();

        if (p->init() == 0) {
            this->active_decode = p;
            std::cout << "DecodeServer init success" << std::endl;
            return 0;
        } else {  // 手动释放
            std::cout << "DecodeServer init current time: " << i << " reconnect.. " << std::endl;
            p->release();
        }
    }
    if (this->active_decode) {
        std::cout << "DecodeServer init success" << std::endl;
        return 0;
    } else {
        std::cerr << "DecodeServer init failed" << std::endl;
        return -1;
    }
}


void DecodeServer::test() {
    if (!this->active_decode) {
        std::cerr << "DecodeServer has not init" << std::endl;
        return;
    } 
    for (int j = 0; j < 2000; ++j) {
        Frame res = this->active_decode->read();
        if (res.ret == SUCCESS) {
            auto img = res.frame;
            std::cout << "main get image height: " << img.rows << "; image width: " << img.cols << std::endl;
        } else {
            j--;  // 这样能保证读取 100 次，查看日志印证了这点
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}