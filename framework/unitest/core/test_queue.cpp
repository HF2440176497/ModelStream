
/**
 * 测试线程安全任务队列
 */

#include <thread>
#include <atomic>

#include "base.hpp"
#include "util/cnstream_queue.hpp"

namespace cnstream {

TEST(TestJobQueue, Basic) {
  ThreadSafeJobQueue<int> q(10, 5);
  q.emplace(1);
  q.emplace(2);
  q.emplace(3);
  EXPECT_EQ(q.qsize(), 3);

  std::vector<int> fetch_jobs {};
  auto ret = q.get_jobs_and_wait(fetch_jobs, 3);
  if (ret) {
    EXPECT_EQ(fetch_jobs.size(), 3);
    EXPECT_EQ(fetch_jobs[0], 1);
    EXPECT_EQ(fetch_jobs[1], 2);
    EXPECT_EQ(fetch_jobs[2], 3);
    EXPECT_EQ(q.qsize(), 0);
  } else {
    LOGE(TestJobQueue) << "fetch_jobs error";
  }

}

TEST(TestJobQueue, BatchEmplace) {
  ThreadSafeJobQueue<int> q(10, 5);
  std::vector<int>        jobs = {1, 2, 3, 4, 5};
  q.batch_emplace(jobs.begin(), jobs.end());
  EXPECT_EQ(q.qsize(), 5);
}

/**
 * 测试多线程环境下的读取
 * 若干进程存入队列，会触发告警；同时若干线程取出队列数据，但是不会检查设置告警标志
 */
TEST(TestJobQueue, Close) {
  ThreadSafeJobQueue<int> job_queue(50, 10);
  job_queue.set_warn_callback([](size_t size) {
    std::cout << "ThreadSafeJobQueue Warn: queue size is " << size << std::endl;
  });
  int total_get_size = 0;
  int target_size = 10;
  std::atomic<bool> produce_done {false};
  std::thread consumer_thread([&job_queue, &total_get_size, &target_size]() {
    std::vector<int> consumed_jobs;  // 总共获取的元素，需要拿到 get_size
    
    // PS: 获取到 target_size 个元素后，才会退出循环
    while (total_get_size < target_size) {
      std::vector<int> jobs;
      bool ret = job_queue.get_jobs_and_wait(jobs, target_size - total_get_size);
      if (ret && !jobs.empty()) {
        consumed_jobs.insert(consumed_jobs.end(), jobs.begin(), jobs.end());
        total_get_size += jobs.size();
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
    }  // end_while
  });

  std::thread producer_thread([&job_queue]() {
    for (int i = 0; i < 15; ++i) {
      job_queue.emplace(i);
    }
  });
  producer_thread.join();
  consumer_thread.join();
}

// 预估现象：
// 因为是两个线程交替运行，produce total num == 15 > warn size;
// 触发 warn callback 的情况无法预估



}