

#include "base.hpp"

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    google::ShutdownGoogleLogging();
    return ret;
}
