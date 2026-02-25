
#ifndef _DECODE_
#define _DECODE_

#include <unistd.h>
#include <signal.h>

#include <memory>
#include <cstdio>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <condition_variable>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>

#ifdef __cplusplus
}
#endif

#include "utils.h"
#include "frame_server.h"


#define SLEEPTIME 60
#define END       1

#define RUN  1
#define STOP 0

#define MAX_QUEUE_SIZE 50

#define RTSP 1
#define RTMP 2
#define ONVIF 3
#define GB 4
#define FILE 5


class Decode {
public:
    Decode(StreamInfo in_stream);
    ~Decode();

private:
    int input_format_init();
    bool support_hwdevice();
    int codec_init();
    int hw_decoder_init();
    int convert_frame_init();
    int init_hwdevice_conf();
    int decode_write();

public:
    int init();
    Frame read();
    void release(int timeout_seconds = 10);
    void get_frame();
    void clean_up();
    void set_run();
    void set_finish();
    void save_image();

public:
    std::atomic<bool> run_flag{true};  // true 表示允许运行; 
    std::atomic<bool> is_run{false};
    std::atomic<bool> init_success{false};  // 表示是否初始化完成
    std::atomic<bool> is_finish{false};  // 表示是否初始化完成
    // std::unique_ptr<std::thread> m_thread;
    
public:
    StreamInfo in_stream;
    AVFormatContext *ifmt_ctx = nullptr;
    AVDictionary *ifmt_opts = nullptr;

private:
    int video_index = -1;
    AVFrame *s_frame = nullptr;  // 最后的输入视频帧
    AVFrame *cv_frame = nullptr;  // 需要提前分配
    uint8_t *cv_buffer = nullptr;  // 存放目标图像的空间

private:
    std::string type_name = "cuda";  // 默认使用 GPU 硬解码
    static std::unordered_map<enum AVCodecID, std::string> codeid_name_table;
    enum AVHWDeviceType device_type = AV_HWDEVICE_TYPE_NONE;
    AVBufferRef *hw_device_ctx = nullptr;

private:
    FrameServer frame_server;
    AVCodec *codec = nullptr;
    AVCodecContext *codec_ctx = nullptr;
    AVCodecParameters *codecpar = nullptr;
    AVPacket pkt;
    struct SwsContext *sws_ctx = nullptr;
private:
    std::mutex mtx_;
    std::condition_variable cv_;
    std::mutex mtx_finish;
    std::condition_variable cv_finish;
};

#endif
