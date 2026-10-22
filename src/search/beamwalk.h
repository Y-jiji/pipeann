#pragma once
// Included by beam_search.cpp and update/conda.cpp, which provide the index headers.

#include <chrono>
#include <tuple>
#include "utils/tsl/robin_set.h"

namespace pipeann {
  template<typename T, typename TagT>
  template<typename SpecFn, typename VerifyFn>
  void SSDIndex<T, TagT>::beam_search_common(const T *query1, uint32_t mem_L, uint64_t l_search,
                                             uint64_t beam_width, SpecFn is_member_approx,
                                             VerifyFn is_member, std::vector<Neighbor> &full_retset,
                                             QueryStats *stats, InsertContext *insert_ctx,
                                             const std::vector<uint32_t> *seeds) {
    // [1]
    auto diskSearchBegin = std::chrono::high_resolution_clock::now();
    QueryBuffer *query_buf = pop_query_buf(query1);
    void *ctx = reader->get_ctx();
    const T *query = query_buf->aligned_query<T>();
    query_buf->reset();
    T *data_buf = query_buf->coord_scratch<T>();
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);
    char *sector_scratch = query_buf->sector_scratch;
    uint64_t &sector_scratch_idx = query_buf->sector_idx;
    nbr_handler->initialize_query(query, query_buf);
    float *dist_scratch = query_buf->aligned_dist_scratch;
    Timer query_timer, io_timer, cpu_timer;
    std::vector<Neighbor> retset(mem_L + kExpandedNodesFactor * l_search);
    tsl::robin_set<uint64_t> visited(4096);
    full_retset.clear();
    full_retset.reserve(kExpandedNodesFactor * l_search);
    uint64_t coord_buf_idx = 0;
    unsigned cur_list_size = 0;

    // [2]
    auto compute_exact_dist_and_push = [&](const DiskNode<T> &node, const unsigned id) {
      memcpy(data_buf, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, data_buf, (unsigned) aligned_dim);
      if (!is_member(id, node)) {
        return;
      }
      if (insert_ctx != nullptr) {
        if (unlikely(coord_buf_idx >= kExpandedNodesFactor * l_search)) {
          LOG(ERROR) << "Please allocate larger coord_buf.";
          crash();
        }
        T *coord_ptr = insert_ctx->coord_buf + coord_buf_idx * aligned_dim;
        memcpy(coord_ptr, node.coords, meta_.data_dim * sizeof(T));
        insert_ctx->coord_map.insert(std::make_pair(id, coord_ptr));
        coord_buf_idx++;
      }
      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));
    };
    auto compute_and_add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids) {
      nbr_handler->compute_dists(query_buf, node_ids, n_ids);
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size].id = node_ids[i];
        retset[cur_list_size].distance = dist_scratch[i];
        retset[cur_list_size++].flag = true;
        visited.insert(node_ids[i]);
      }
    };

    // [3]
    if (seeds != nullptr) {
      const uint64_t n_seeds = std::min<uint64_t>({seeds->size(), MAX_N_EDGES, retset.size()});
      nbr_handler->compute_dists(query_buf, seeds->data(), n_seeds);
      for (uint64_t i = 0; i < n_seeds; ++i) {
        retset[cur_list_size++] = Neighbor((*seeds)[i], dist_scratch[i], true);
        visited.insert((*seeds)[i]);
      }
    } else if (mem_L) {
      std::vector<unsigned> mem_tags(mem_L);
      std::vector<float> mem_dists(mem_L);
      mem_index_->search_with_tags(query, mem_L, mem_L, mem_tags.data(), mem_dists.data());
      compute_and_add_to_retset(mem_tags.data(), std::min<uint64_t>(mem_L, l_search));
    } else {
      const uint32_t start = this->randentry();
      compute_and_add_to_retset(&start, 1);
    }
    std::sort(retset.begin(), retset.begin() + cur_list_size);
    cur_list_size = std::min<uint64_t>(cur_list_size, l_search);

    // [4]
    unsigned k = 0;
    std::vector<unsigned> frontier;
    using fnhood_t = std::tuple<unsigned, unsigned, char *>;
    std::vector<fnhood_t> frontier_nhoods;
    std::vector<IORequest> frontier_read_reqs;
    std::vector<uint64_t> page_ref;
    auto *page_ref_out = insert_ctx != nullptr ? &insert_ctx->page_ref : &page_ref;
    while (k < cur_list_size) {
      auto nk = cur_list_size;
      frontier.clear();
      frontier_nhoods.clear();
      frontier_read_reqs.clear();
      sector_scratch_idx = 0;
      uint32_t marker = k;
      uint32_t num_seen = 0;
      while (marker < cur_list_size && frontier.size() < beam_width && num_seen < beam_width) {
        if (retset[marker].flag) {
          num_seen++;
          frontier.push_back(retset[marker].id);
          retset[marker].flag = false;
        }
        marker++;
      }

      // [5]
      if (!frontier.empty()) {
        if (stats != nullptr) {
          stats->n_hops++;
        }
        for (uint64_t i = 0; i < frontier.size(); i++) {
          uint32_t id = frontier[i];
          uint32_t loc = this->id2loc(id);
          uint64_t offset = loc_sector_no(loc) * SECTOR_LEN;
          if (insert_ctx != nullptr) {
            insert_ctx->hint_pages.push_back(offset / SECTOR_LEN);
          }
          auto sector_buf = sector_scratch + sector_scratch_idx * io_size;
          frontier_nhoods.emplace_back(id, loc, sector_buf);
          sector_scratch_idx++;
          frontier_read_reqs.emplace_back(IORequest(offset, io_size, sector_buf, u_loc_offset(loc),
                                                    meta_.max_node_len, sector_scratch));
          if (stats != nullptr) {
            stats->n_ios++;
          }
        }
        io_timer.reset();
        reader->read_alloc(frontier_read_reqs, ctx, page_ref_out);
        if (stats != nullptr) {
          stats->io_us += (double) io_timer.elapsed();
        }
      }

      // [6]
      for (auto &frontier_nhood : frontier_nhoods) {
        auto [id, loc, sector_buf] = frontier_nhood;
        DiskNode<T> node = node_from_page(sector_buf, loc);
        compute_exact_dist_and_push(node, id);
        cpu_timer.reset();
        nbr_handler->compute_dists(query_buf, node.nbrs, node.nnbrs);
        if (stats != nullptr) {
          stats->n_cmps += (double) node.nnbrs;
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
        cpu_timer.reset();
        for (uint64_t m = 0; m < node.nnbrs; ++m) {
          unsigned id = node.nbrs[m];
          if (unlikely(id > this->cur_id)) {
            LOG(ERROR) << "ID is larger than current ID, " << id << " vs " << this->cur_id;
            crash();
          }
          if (visited.find(id) != visited.end() || !is_member_approx(id)) {
            continue;
          }
          visited.insert(id);
          float dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }
          if (dist >= retset[cur_list_size - 1].distance && (cur_list_size == l_search)) {
            continue;
          }
          Neighbor nn(id, dist, true);
          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_search) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk) {
            nk = r;
          }
        }
        if (stats != nullptr) {
          stats->cpu_us += (double) cpu_timer.elapsed();
        }
      }

      // [7]
      if (nk <= k) {
        k = nk;
      } else {
        ++k;
      }
      if (stats != nullptr && stats->n_current_used != 0) {
        auto diskSearchEnd = std::chrono::high_resolution_clock::now();
        auto elapsed = diskSearchEnd - diskSearchBegin;
        double elapsedSeconds =
            std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        if (elapsedSeconds >= stats->n_current_used) {
          break;
        }
      }
    }

    // [8]
    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });
    if (insert_ctx == nullptr) {
      reader->deref(&page_ref);
    }
    push_query_buf(query_buf);
    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }
}  // namespace pipeann
