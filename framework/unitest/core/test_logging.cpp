
#include <gtest/gtest.h>

#include "base.hpp"
#include "cnstream_logging.hpp"


/**
 * @brief Test logging
 */
TEST(Logging, TestLog) {
    LOGI(COREUNITEST) << "Test log info";
    LOGW(COREUNITEST) << "Test log warning";
    LOGE(COREUNITEST) << "Test log error";
}