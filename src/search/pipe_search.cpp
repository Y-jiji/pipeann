#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <queue>
#include <unordered_map>

#include "utils/timer.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "pipe_search_common.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_pipe_search(const T *query, uint32_t mem_L, uint32_t l_search, const uint64_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats,
                                         InsertContext *insert_ctx) {
    auto always_member = [](unsigned) -> bool { return true; };
    auto always_valid = [](unsigned, const DiskNode<T> &) -> bool { return true; };

    this->pipe_search_common(query, mem_L, l_search, l_search, beam_width, io_size, false, always_member, always_valid,
                             expanded_nodes_info, stats, insert_ctx);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    if (params.conda != 0) {
      auto untombed = [this](unsigned id) -> bool {
        return id >= condahead.size() || condahead[id] != 2;
      };
      auto always_valid = [](unsigned, const DiskNode<T> &) -> bool { return true; };
      this->pipe_search_common(query, mem_L, l_search, l_search, beam_width, io_size, false, untombed,
                               always_valid, expanded_nodes_info, stats, nullptr);
    } else {
      this->do_pipe_search(query, mem_L, (uint32_t) l_search, beam_width, expanded_nodes_info, stats, nullptr);
    }
    return copy_top_k(expanded_nodes_info, k_search, res_tags, distances);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::range_search(const T *query, const float range, TagT *res_tags, float *res_dists,
                                         const uint64_t beam_width, const uint32_t mem_L, const uint64_t l_search,
                                         QueryStats *stats) {
    std::shared_lock lk(merge_lock);
    auto always_member = [](unsigned) -> bool { return true; };
    auto always_valid = [](unsigned, const DiskNode<T> &) -> bool { return true; };

    const float range_partial = get_partial_order_distance<T>(range, this->metric);
    std::vector<Neighbor> full_retset;
    this->pipe_search_common(query, mem_L, l_search, l_search, beam_width, io_size, false, always_member, always_valid,
                             full_retset, stats, nullptr, range_partial);
    return copy_top_k(full_retset, l_search, res_tags, res_dists, range_partial);
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
