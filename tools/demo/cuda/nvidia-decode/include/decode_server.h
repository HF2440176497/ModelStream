
#ifndef _DECODE_SERVER_
#define _DECODE_SERVER_

#include <vector>
#include <memory>

#include "decode.h"


class DecodeServer 
{
public:
    DecodeServer(StreamInfo in_stream);
    ~DecodeServer();

public:
    int init();
    const int max_reconnect_num = 5;

public:
    void test();

public:
    StreamInfo in_stream;
    FrameServer frameserver;
    std::vector<std::shared_ptr<Decode>> decoders {};
    std::shared_ptr<Decode> active_decode = nullptr;
};


#endif