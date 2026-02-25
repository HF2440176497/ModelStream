
#include <memory>
#include <unordered_map>

#include "decode.h"
#include "frame_server.h"
#include "utils.h"

static int save_count = 10;
static enum AVPixelFormat hw_pix_fmt;

std::unordered_map<enum AVCodecID, std::string> Decode::codeid_name_table = {
    {AV_CODEC_ID_H264, "h264_cuvid"},
    {AV_CODEC_ID_HEVC, "hevc_cuvid"},
    {AV_CODEC_ID_VP8, "vp8_cuvid"},
    {AV_CODEC_ID_VP9, "vp9_cuvid"},
    {AV_CODEC_ID_AV1, "av1_cuvid"},
};

void print_all_available_device() {
    enum AVHWDeviceType type = AV_HWDEVICE_TYPE_NONE;
    std::cout << "Available device types:" << std::endl;
    while((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE) {
        printf("supported hardware device type: %s\n", av_hwdevice_get_type_name(type));
    }
    std::cout << std::endl;
}

Decode::Decode(StreamInfo in_stream) : run_flag(true), in_stream(in_stream), frame_server() {}

Decode::~Decode() {
    this->clean_up();
}

/**
 * 如果
 */
int Decode::init() { 
    bool status;
    {
        std::unique_lock<std::mutex> lck(mtx_);
        auto timeout = std::chrono::seconds(10); // 设置超时时间为5秒
        status = cv_.wait_for(lck, timeout, [this] { 
            return this->is_run && (!this->is_finish); 
        });  // 开始运行
    }
    {
        std::unique_lock<std::mutex> lck(mtx_);
        auto timeout = std::chrono::seconds(10); // 设置超时时间为5秒
        status = cv_.wait_for(lck, timeout, [this] { 
            return this->init_success && (!this->is_finish); 
        });  // 初始化还未结束
    }
    // 如果已经 finish, 那么一定说明 is_run 或者 init 设置过
    if (this->is_finish) {
        std::cout << "Decode init: thread has finish, init exits" << std::endl;
        return -1;
    }
    if (is_run && init_success) {
        return 0;
    }
    std::cout << "Decode init: stick: " << is_run << "; " << init_success << std::endl;
    return -1;
}

Frame Decode::read() {
    Frame frame = this->frame_server.get();
    return frame;
}

/**
 * 外界调用
 */
void Decode::release(int timeout_seconds) {
    this->run_flag = false;
    {
        std::unique_lock<std::mutex> lck(mtx_);
        auto timeout = std::chrono::seconds(timeout_seconds);
        bool status = cv_.wait_for(lck, timeout, [this] { 
            return this->is_run || this->init_success;
        });
    }
    {
        std::unique_lock<std::mutex> lck(mtx_);
        auto timeout = std::chrono::seconds(timeout_seconds);
        bool status = cv_.wait_for(lck, timeout, [this] { 
            return !this->is_finish;
        });
    }
    std::cout << "Decode::release: " << this->is_finish << std::endl;
}

/**
 * 现象：
 */

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx, const enum AVPixelFormat *pix_fmts) {
    const enum AVPixelFormat *p;
    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt) {
            return *p;  // 当前支持格式，
        }
    }
    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}

int Decode::codec_init() {
    int ret = 0;
    AVStream* video_stream = this->ifmt_ctx->streams[this->video_index];
    auto it = this->codeid_name_table.find(video_stream->codecpar->codec_id);
    if (it == this->codeid_name_table.end()) {
        std::cerr << "Codec name not found" << std::endl;
        return -1;
    }
    std::string decoder_name = it->second;
    this->codec = const_cast<AVCodec*>(avcodec_find_decoder_by_name(decoder_name.c_str()));
    if (!this->codec) {
        std::cerr << "Codec not found" << std::endl;
        return -1;
    }
    if ((ret = this->init_hwdevice_conf()) != 0) {
        std::cerr << "init_hwdevice_conf error" << std::endl;
        return ret;
    }
    if (!(this->codec)) {
        std::cerr << "Warn: codec is null" << std::endl;
    }
    // 临时测试
    if (!(this->codec_ctx = avcodec_alloc_context3(this->codec))) {
        std::cerr << "avcodec_alloc_context error" << std::endl;
        return -1;
    }
    if ((ret = avcodec_parameters_to_context(this->codec_ctx, video_stream->codecpar)) < 0) {
        std::cerr << "avcodec_parameters_to_context error" << std::endl;
        return ret;
    }
    this->codec_ctx->pkt_timebase.num = video_stream->time_base.num; // 我的是加了这里才正常
    this->codec_ctx->pkt_timebase.den = video_stream->time_base.den; // 我的是加了这里才正常
    this->codec_ctx->get_format = get_hw_format;  // 回调函数
    if ((ret = this->hw_decoder_init()) < 0) {
        std::cerr << "hw_decoder_init error" << std::endl;
        return ret;
    }
    // 绑定完成后 打开编解码器
    if ((ret = avcodec_open2(this->codec_ctx, this->codec, NULL)) < 0) {
        std::cerr << "Failed to open codec for stream" << std::endl;
        return ret;
    }
    return ret;
}

// 创建硬件上下文，并配置到编解码器
int Decode::hw_decoder_init() {
    int err = 0;
    if (this->hw_device_ctx) {  // 此时应当为空
        std::cerr << "hw_device_ctx should be null" << std::endl;
        return -1;
    }
    if ((err = av_hwdevice_ctx_create(&(this->hw_device_ctx), this->device_type, NULL, NULL, 0)) < 0) {
        std::cerr << "Failed to create specified HW device" << std::endl;
        return err;
    }
    this->codec_ctx->hw_device_ctx = av_buffer_ref(this->hw_device_ctx);
    return err;
}

int Decode::input_format_init() {
    int ret = 0;
    if ((ret = avformat_network_init()) != 0) {
        std::cout << "avformat_network_init failed: " << ret << std::endl;
        return ret;
    }
    this->ifmt_ctx = avformat_alloc_context();
    if (!(this->ifmt_ctx)) {
        std::cerr << "avformat_find_stream_info error" << std::endl;
        return -1;
    }
    // 设置相关参数
    auto url_cstr = this->in_stream.url.c_str();
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "buffer_size", "1024000", 0);
    av_dict_set(&opts, "max_delay", "200000", 0);
    av_dict_set(&opts, "stimeout", "20000000", 0);
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    ret = avformat_open_input(&(this->ifmt_ctx), url_cstr, NULL, &opts);
    if (ret != 0) {
        std::cerr << "avformat_open_input error" << std::endl;
        av_dict_free(&opts);
        return ret;
    }
    av_dict_free(&opts);
    ret = avformat_find_stream_info(this->ifmt_ctx, nullptr);
    if (ret < 0) {
        return ret;
    }
    // 视频流及匹配的编码器
    std::cout << "this->ifmt_ctx->nb_streams: " << this->ifmt_ctx->nb_streams << std::endl;
    for (unsigned int i = 0; i < this->ifmt_ctx->nb_streams; ++i) {
        AVCodecParameters* codec_par = this->ifmt_ctx->streams[i]->codecpar;
        if (codec_par->codec_type == AVMEDIA_TYPE_VIDEO) {
            this->video_index = i;
            break;
        }
    }
    if (this->video_index < 0) {
        std::cerr << "av_find_stream error" << std::endl;
        return -1;
    }
    return ret;
}


/**
 * 打印所有支持的设备
 */
bool Decode::support_hwdevice() {
    print_all_available_device();
    enum AVHWDeviceType type = av_hwdevice_find_type_by_name(this->type_name.c_str());
    if (type == AV_HWDEVICE_TYPE_NONE) {
        std::cerr << "Device type: " << this->type_name << " is not supported." << std::endl;
        return false;
    }
    this->device_type = type;
    return true;
}


int Decode::decode_write() {
    int ret = 0;
    AVFrame *p_frame = nullptr;
    AVFrame *sw_frame = nullptr;
    uint8_t *buffer = nullptr;  // 保存解码得到的

    ret = avcodec_send_packet(this->codec_ctx, &(this->pkt));
    if (ret < 0) {
        std::cerr << "avcodec_send_packet" << std::endl;
        return ret;
    }
    while (this->run_flag) {
        // std::cout << "Decode::decode_write: enter to while" << std::endl;
        if (!(p_frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())) {
            std::cerr << "av_frame_alloc error" << std::endl;
            ret = -1;
            break;  // 表示释放资源
        }
        ret = avcodec_receive_frame(this->codec_ctx, p_frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&p_frame);
            av_frame_free(&sw_frame);
            return 0;  // 表示正常结束
        } else if (ret < 0) {
            std::cerr << "Error during decoding" << std::endl;
            break;
        } else {
            
        }
        std::cout << "Decode::decode_write: receive frame" << std::endl;
		if (p_frame->format == hw_pix_fmt) {
			/* retrieve data from GPU to CPU */
			std::cout << "retrieve data from GPU to CPU" << std::endl;
			if ((ret = av_hwframe_transfer_data(sw_frame, p_frame, 0)) < 0) {
				std::cerr << "Error transferring the data to system memory" << std::endl;
                break;
			}
			this->s_frame = sw_frame;
		} else {
            this->s_frame = p_frame;
        }
        if (!(this->s_frame)) {
            std::cerr << "s_frame is null" << std::endl;
            ret = -1;
            break;
        }
	/*
		int size = av_image_get_buffer_size((AVPixelFormat)this->s_frame->format, this->s_frame->width,
			this->s_frame->height, 1);
		buffer = (uint8_t *)av_malloc(size);
		if (!buffer) {
			fprintf(stderr, "Can not alloc buffer\n");
            ret = -1;
			break;
		}
		std::cout << "s_frame pix_fmt: " << (AVPixelFormat)this->s_frame->format << "; " << AV_PIX_FMT_YUV420P << std::endl;
		ret = av_image_copy_to_buffer(buffer, size,
			(const uint8_t * const *)this->s_frame->data,
			(const int *)this->s_frame->linesize, (AVPixelFormat)this->s_frame->format,
			this->s_frame->width, this->s_frame->height, 1);
		if (ret < 0) {
			fprintf(stderr, "Can not copy image to buffer\n");
			break;
		}
	*/
        auto dst_width = this->codec_ctx->width;
        auto dst_height = this->codec_ctx->height;
        enum AVPixelFormat sw_format = (enum AVPixelFormat)this->s_frame->format; 
        this->sws_ctx = sws_getCachedContext(this->sws_ctx, this->codec_ctx->width, this->codec_ctx->height, sw_format, 
            this->s_frame->width, this->s_frame->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
        if (!(this->sws_ctx)) {
            std::cerr << "Could not allocate SwsContext" << std::endl;
            ret = -1;
            break;
        }
        // enum AVPixelFormat sw_format = (enum AVPixelFormat)this->s_frame->format;
        // // std::cout << "sw_format: " << typeid(sw_format).name() << ": " << sw_format << std::endl;
        // std::cout << "this->s_frame->height: " << this->s_frame->height << std::endl;
        // std::cout << "this->s_frame->width: " << this->s_frame->width << std::endl;    
        // this->sws_ctx = sws_getCachedContext(this->s_frame->width, this->s_frame->height, sw_format, this->s_frame->width, this->s_frame->height, AV_PIX_FMT_BGR24, SWS_BILINEAR, NULL, NULL, NULL);
        // if (!(this->sws_ctx)) {
        //     std::cerr << "Could not allocate SwsContext" << std::endl;
        //     ret = -1;
        //     break;
        // }
        ret = sws_scale(this->sws_ctx, (const uint8_t *const *)this->s_frame->data, this->s_frame->linesize, 0,
                this->s_frame->height, this->cv_frame->data, this->cv_frame->linesize);
        if (ret < 0) {
            std::cerr << "sws_scale error" << std::endl;
            break;
        }
        this->save_image();
        std::cout << "save_image complete" << std::endl;
        
        av_frame_free(&p_frame);
        av_frame_free(&sw_frame);
        if (buffer) {
            av_freep(&buffer);
        }
    }  // end while(true)
    av_frame_free(&p_frame);
    av_frame_free(&sw_frame);
    if (buffer) {
        av_freep(&buffer);
    }
    return ret;
}

/**
 * 
 */
void Decode::set_finish() {
    {
        std::lock_guard<std::mutex> lck(mtx_finish);
        this->is_finish = true;
    }
    this->cv_finish.notify_all();
}

void Decode::set_run() {
    {
        std::lock_guard<std::mutex> lck(mtx_);
        this->is_run = true;
    }
    this->cv_.notify_all();  // 通知主线程 
}

/**
 * 在外使用：先启动 get_frame 线程，再调用 init 初始化
 * Decode::init 相当于返回初始化完成
 */
void Decode::get_frame() {
    int ret = 0;
    //av_register_all();
    this->set_run();
    std::cout << "Decode::get_frame: get_frame start..." << std::endl;
    if (!(this->run_flag) || !this->support_hwdevice()) {
        std::cerr << "support_hwdevice return exit" << std::endl;
        this->set_finish();
        return;
    }
    if (!(this->run_flag) || (ret = this->input_format_init()) < 0) {
        std::cerr << "this->input_format_init exit" << std::endl;
        this->set_finish();
        return;
    }
    std::cout << "Decode::get_frame: input_format_init complete" << std::endl;
    if (!(this->run_flag) || (ret = this->codec_init()) < 0) {
        std::cerr << "this->codec_init exit" << std::endl;
        this->set_finish();
        return;
    }
    std::cout << "Decode::get_frame: codec_init complete" << std::endl;
    if (!(this->run_flag) || (ret = this->convert_frame_init()) < 0) {
        std::cerr << "this->convert_frame_init exit" << std::endl;
        this->set_finish();
        return;
    }
    while (this->run_flag && ret >= 0) {
        if (!(this->init_success)) {
            {
                std::lock_guard<std::mutex> lck(mtx_);
                this->init_success = true;  // 初始化完成
            }
            cv_.notify_all();
        }
        std::cout << "Decode::get_frame: enter to while" << std::endl;
        ret = av_read_frame(this->ifmt_ctx, &(this->pkt));
        if (ret < 0) {
            std::cout << "Decode::get_frame: av_read_frame error" << std::endl;
            break;
        }
        if (this->pkt.stream_index != this->video_index) {
            av_packet_unref(&(this->pkt));  // 重置 AVPacket
            continue;
        }
        if ((ret = this->decode_write()) < 0) {
            break;
        }
        av_packet_unref(&(this->pkt));
    }  // end thread while
    this->set_finish();
}


/**
 * 在循环开始前设置转换后的目标帧
 */
int Decode::convert_frame_init() {
    int ret = 0;
    if (!(this->cv_frame = av_frame_alloc())) {
        std::cerr << "cv_frame av_frame_alloc error" << std::endl;
        return -1;
    }
    // 分配、设置目标帧
    auto dst_width = this->codec_ctx->width;
    auto dst_height = this->codec_ctx->height;
    auto dst_pix_fmt = AV_PIX_FMT_BGR24;
    auto dst_img_size = av_image_get_buffer_size(dst_pix_fmt, dst_width, dst_height, 1);
    // 手动分配空间
    this->cv_buffer = (uint8_t*)av_malloc(dst_img_size *sizeof(uint8_t));
    av_image_fill_arrays(this->cv_frame->data, this->cv_frame->linesize, this->cv_buffer, dst_pix_fmt, dst_width,
                         dst_height, 1);
    return ret;
}


/**
 * 遍历硬件 ID 获取编解码器支持的
 */
int Decode::init_hwdevice_conf() {
    for (int i = 0;; i++) {
        const AVCodecHWConfig *config =
            avcodec_get_hw_config(this->codec, i);  // 用于获取编解码器支持的硬件配置AVCodecHWConfig
        if (!config) {
            fprintf(stderr, "Decoder %s does not support device type %s.\n", this->codec->name,
                    av_hwdevice_get_type_name(this->device_type));
            return -1;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX && config->device_type == this->device_type) {
            std::cout << "Decoder " << this->codec->name << " supports device type: " << av_hwdevice_get_type_name(this->device_type) 
                    << " with pix_fmt: " << av_get_pix_fmt_name(config->pix_fmt) << std::endl;
            if (config->pix_fmt == AV_PIX_FMT_CUDA) {
                std::cout << " supports CUDA pix_fmt: " << av_get_pix_fmt_name(config->pix_fmt) << std::endl;
            }
            hw_pix_fmt = config->pix_fmt;  // 设置为硬件支持的像素格式
            return 0;
        }
    }
    return -1;
}

void Decode::clean_up() {
    av_frame_free(&(this->s_frame));
    av_frame_free(&(this->cv_frame));
    if (this->sws_ctx) {
        sws_freeContext(this->sws_ctx);
    }
    if (this->codec_ctx) {
        avcodec_free_context(&(this->codec_ctx));
    }
    if (this->ifmt_ctx) {
        avformat_close_input(&(this->ifmt_ctx));
    }
    if (this->hw_device_ctx) {
        av_buffer_unref(&(this->hw_device_ctx));
    }
}

void Decode::save_image() {
    Frame cur_frame = Frame();
    cv::Mat cv_mat(this->s_frame->height, this->s_frame->width, CV_8UC3, this->cv_buffer);
    cur_frame.frame = cv_mat.clone();
    int rows = cur_frame.frame.rows;
    int cols = cur_frame.frame.cols;
    int channels = cur_frame.frame.channels();
    // std::cout << "Image shape: (" << rows << ", " << cols << ", " << channels << ")" << std::endl;

    // if (save_count--) {
    //     std::string file_name = std::to_string(get_timestamp_ms()) + ".png";
    //     bool result = cv::imwrite(file_name, *(cur_frame.p_frame));
    //     if (!result) {
    //         std::cerr << "image save error" << std::endl;
    //     }
    // }

    cur_frame.ret = SUCCESS;  // 表示成功
    this->frame_server.put(cur_frame);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    return;
}
