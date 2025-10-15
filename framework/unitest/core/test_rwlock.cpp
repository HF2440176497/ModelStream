
#include <gtest/gtest.h>
#include "util/cnstream_rwlock.hpp"

#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
/**
 * 
 */

namespace cnstream {

/**
 * @brief 测试读写锁的基本功能
 */
class RwLockTest : public ::testing::Test {
protected:
    void SetUp() override {
        shared_data = 0;
        read_count = 0;
        write_count = 0;
    }
    void TearDown() override {}

    RwLock rwlock;
    int shared_data;  // 共享数据
    std::atomic<int> read_count;
    std::atomic<int> write_count;
};


TEST_F(RwLockTest, BasicReadWrite) {
    {
        RwLockWriteGuard guard(rwlock);
        shared_data = 100;
    }
    {
        RwLockReadGuard guard(rwlock);
        EXPECT_EQ(shared_data, 100);
    }
}

/**
 * @brief 测试多个写入器同时写入共享数据
 */
TEST_F(RwLockTest, ConcurrentWrite) {
    RwLock lock;
    int data = 0;
    std::atomic<int> active_writers{0};

    auto writer = [&](int val) {
        // 各线程串行：比较-加1-减1 是一个完整操作
        RwLockWriteGuard wg(lock);
        EXPECT_EQ(active_writers.fetch_add(1), 0) << "write lock not exclusive";
        data = val;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        active_writers.fetch_sub(1);
    };
    std::vector<std::thread> ths;
    for (int i = 1; i <= 4; ++i) ths.emplace_back(writer, i);
    for (auto& t : ths) t.join();
    EXPECT_NE(data, 0);   // 不等于 证明至少有一次写入成功
}

// 测试读写互斥
TEST_F(RwLockTest, ReadWriteExclusion) {
    std::atomic<bool> reader_started(false);
    std::atomic<bool> writer_started(false);
    std::atomic<bool> read_during_write(false);
    std::atomic<bool> write_during_read(false);
    
    std::thread writer([&]() {
        RwLockWriteGuard guard(rwlock);
        writer_started = true;   
        // 等待读者尝试读取 读者运行到 RwLockReadGuard 会阻塞
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        // 检查在写锁持有期间是否有读者进入
        // 不应该通过此判断，因为读者只有在 writer 退出后才有机会 reader_started = true
        if (reader_started) {
            read_during_write = true;
        }
    });
    
    std::thread reader([&]() {
        // 等待写者先获取锁
        while (!writer_started) {
            std::this_thread::yield();
        }
        // 尝试获取读锁（应该被阻塞直到写锁释放）
        RwLockReadGuard guard(rwlock);
        reader_started = true;
    });
    
    writer.join();
    reader.join();
    
    EXPECT_FALSE(read_during_write);
    EXPECT_TRUE(reader_started);
}

}
