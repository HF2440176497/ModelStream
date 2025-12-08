

/**
 * 全局常量
 */


#ifndef CNSTREAM_CONSTANTS_PRI_HPP_
#define CNSTREAM_CONSTANTS_PRI_HPP_

#define CNS_JSON_DIR_PARAM_NAME "json_file_dir"

inline constexpr size_t INVALID_MODULE_ID = (size_t)(-1);
inline constexpr uint32_t INVALID_STREAM_IDX = (uint32_t)(-1);
inline constexpr uint32_t MAX_STREAM_NUM = 128; /*!< The streams at most allowed. */

/**
 * @brief Profiler configuration title in JSON configuration file.
 **/
inline constexpr char kProfilerConfigName[] = "profiler_config";

/**
 * @brief Subgraph node item prefix.
 **/
inline constexpr char kSubgraphConfigPrefix[] = "subgraph:";

#endif