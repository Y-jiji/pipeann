#include "aligned_file_reader.h"
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "ssd_index.h"
#include <malloc.h>
#include <algorithm>

#include <omp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include "utils/timer.h"
#include "utils.h"
#include "utils/page_cache.h"

#include <unistd.h>
#include <sys/syscall.h>
#include "linux_aligned_file_reader.h"
#include "beamwalk.h"

namespace pipeann {
  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::do_beam_search(const T *query1, uint32_t mem_L, uint32_t l_search, const uint64_t beam_width,
                                         std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats,
                                         InsertContext *insert_ctx) {
    auto always_member = [](unsigned) -> bool { return true; };
    auto always_valid = [](unsigned, const DiskNode<T> &) -> bool { return true; };
    this->beam_search_common(query1, mem_L, l_search, beam_width, always_member, always_valid,
                             expanded_nodes_info, stats, insert_ctx);
  }

  template<typename T, typename TagT>
  size_t SSDIndex<T, TagT>::beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L,
                                        const uint64_t l_search, TagT *res_tags, float *distances,
                                        const uint64_t beam_width, QueryStats *stats) {
    std::shared_lock lk(merge_lock);
    std::vector<Neighbor> expanded_nodes_info;
    this->do_beam_search(query, mem_L, (uint32_t) l_search, beam_width, expanded_nodes_info, stats, nullptr);

    return copy_top_k(expanded_nodes_info, k_search, res_tags, distances);
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
