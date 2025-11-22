
/**
 * Unit test for Connector and conveyor
 */


#include "cnstream_connector.hpp"
#include "cnstream_conveyor.hpp"
#include "cnstream_frame.hpp"

#include "base.hpp"


namespace cnstream {


class ConnectorTest : public testing::Test {
 protected:
  ConnectorTest() = default;
  void SetUp() override {
    connector_ = std::make_unique<Connector>(conveyor_count_, max_datas_num);
    connector_->Start();
  }

  void TearDown() override {
    connector_->Stop();
  }
 protected:
  const int conveyor_count_ = 20;
  const int max_datas_num = 100;
  std::unique_ptr<Connector> connector_;
};


TEST_F(ConnectorTest, MultiThread) {
  std::vector<std::thread> producers;
  std::vector<std::thread> consumers;

  producers.reserve(conveyor_count_);
  consumers.reserve(conveyor_count_);

  const int data_nums = 80;  // 放入数据的数量
  for (int i = 0; i < conveyor_count_; ++i) {
    producers.emplace_back([this, i, data_nums]() {
      for (int j = 0; j < data_nums; ++j) {
        CNFrameInfoPtr frame = CNFrameInfo::Create("test-stream-" + std::to_string(i));
        frame->test_idx = i;  // thread 对应 idx, 因此每个 conveyor 内的 frame->idx 也相同
        while (connector_->IsRunning() && (connector_->PushDataBufferToConveyor(i, frame) == false)) {
          // std::cout << "producer: " << i << " conveyor size: " << connector_->GetConveyorSize(i) << std::endl;
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
    });
    consumers.emplace_back([this, i, data_nums]() {
      // 每个线程从对应的 conveyor 获取放入的 frame
      for (int j = 0; j < data_nums; ++j) {
        CNFrameInfoPtr get_frame = nullptr;
        // 仿照 Pipeline 的做法
        while (connector_->IsRunning() && (get_frame == nullptr)) {
          get_frame = connector_->PopDataBufferFromConveyor(i);
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (get_frame) {  // 需要此判断
          EXPECT_EQ(get_frame->test_idx, i);
        }
      }
    });
  }

  for (int i = 0; i < conveyor_count_; i++) {
    producers[i].join();
    consumers[i].join();
  }
}

}  // end namespace cnstream



