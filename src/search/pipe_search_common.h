#pragma once
// Included by pipe_search.cpp and spec_filter_search.cpp which provide all necessary headers.

#ifdef USE_URING
#include "liburing.h"
#endif

namespace pipeann {

  // Common pipelined search loop used by pipe_search, spec_postfilter_search, and spec_infilter_search.
  //
  // Template callbacks:
  //   SpecFn: (unsigned id) -> bool
  //     Approximate membership check for neighbor filtering during graph traversal.
  //     pipe_search / post_filter: always return true.
  //     in_filter: selector->is_member_approx(id, query_attrs).
  //
  //   VerifyFn: (unsigned id, const DiskNode<T>& node) -> bool
  //     Exact membership verification after reading a node from disk.
  //     pipe_search: always return true.
  //     filter search: selector->is_member(id, query_attrs, target_attrs).
  //
  template<typename T, typename TagT>
  template<typename SpecFn, typename VerifyFn>
  void SSDIndex<T, TagT>::pipe_search_common(const T *query1, uint32_t mem_L, uint64_t l_search, uint64_t l_pool,
                                             uint64_t beam_width, uint64_t read_io_size, bool use_dense_nbrs,
                                             SpecFn is_member_approx, VerifyFn is_member,
                                             std::vector<Neighbor> &full_retset, QueryStats *stats,
                                             InsertContext *insert_ctx, float range_partial) {
    QueryBuffer *query_buf = pop_query_buf(query1);
#ifdef USE_URING
    void *ctx = reader->get_ctx(IORING_SETUP_SQPOLL);
#else
    void *ctx = reader->get_ctx();
#endif

    if (beam_width > MAX_N_SECTOR_READS) {
      LOG(ERROR) << "Beamwidth can not be higher than MAX_N_SECTOR_READS";
      crash();
    }

    const T *query = query_buf->aligned_query<T>();
    query_buf->reset();

    T *data_buf = query_buf->coord_scratch<T>();
    _mm_prefetch((char *) data_buf, _MM_HINT_T1);

    char *sector_scratch = query_buf->sector_scratch;
    float *dist_scratch = query_buf->aligned_dist_scratch;

    Timer query_timer;
    std::vector<Neighbor> retset(mem_L + this->params.R + l_pool * kExpandedNodesFactor);
    auto &visited = *(query_buf->visited);
    unsigned cur_list_size = 0;

    full_retset.reserve(l_pool * kExpandedNodesFactor);

    // --- Exact distance computation + membership verification ---
    uint64_t coord_buf_idx = 0;
    auto compute_exact_dists_and_push = [&](const DiskNode<T> &node, const unsigned id) -> bool {
      T *node_fp_coords_copy = data_buf;
      memcpy(node_fp_coords_copy, node.coords, meta_.data_dim * sizeof(T));
      float cur_expanded_dist = dist_cmp->compare(query, node_fp_coords_copy, (unsigned) aligned_dim);
      if (stats != nullptr) {
        stats->n_exact++;
      }

      if (!is_member(id, node)) {
        return false;
      }

      full_retset.push_back(Neighbor(id, cur_expanded_dist, true));

      if (insert_ctx != nullptr) {
        if (unlikely(coord_buf_idx >= kExpandedNodesFactor * l_pool)) {
          LOG(ERROR) << "Please allocate larger coord_buf.";
          crash();
        }
        T *coord_ptr = insert_ctx->coord_buf + coord_buf_idx * aligned_dim;
        memcpy(coord_ptr, node.coords, meta_.data_dim * sizeof(T));
        insert_ctx->coord_map.insert(std::make_pair(id, coord_ptr));
        coord_buf_idx++;
      }
      return true;
    };

    // --- Unified neighbor expansion (1-hop approx -> 2-hop approx -> 1-hop connectivity) ---
    uint64_t n_computes = 0;
    auto compute_and_push_nbrs = [&](DiskNode<T> &node, unsigned &nk) {
      unsigned nbors_cand_size = 0;
      uint32_t *nbr_ids = query_buf->nbr_id_scratch;

      // 1-hop neighbors that are approximate members.
      for (unsigned m = 0; m < node.nnbrs && nbors_cand_size < node.nnbrs; ++m) {
        if (visited.find(node.nbrs[m]) == visited.end() && is_member_approx(node.nbrs[m])) {
          nbr_ids[nbors_cand_size++] = node.nbrs[m];
          visited.insert(node.nbrs[m]);
        }
      }

      unsigned n_member = nbors_cand_size;

      // Only valid for in-filter search where dense neighbors were actually read from disk.
      if (use_dense_nbrs) {
        // 2-hop (dense) neighbors that are approximate members.
        for (unsigned m = 0; m < node.n_dense_nbrs && nbors_cand_size < node.nnbrs; ++m) {
          if (visited.find(node.dense_nbrs[m]) == visited.end() && is_member_approx(node.dense_nbrs[m])) {
            nbr_ids[nbors_cand_size++] = node.dense_nbrs[m];
            visited.insert(node.dense_nbrs[m]);
          }
        }
        n_member = nbors_cand_size;
        // Remaining 1-hop neighbors for connectivity.
        for (unsigned m = 0; m < node.nnbrs && nbors_cand_size < node.nnbrs; ++m) {
          if (visited.find(node.nbrs[m]) == visited.end()) {
            nbr_ids[nbors_cand_size++] = node.nbrs[m];
            visited.insert(node.nbrs[m]);
          }
        }
      }

      n_computes += nbors_cand_size;
      if (nbors_cand_size) {
        nbr_handler->compute_dists(query_buf, nbr_ids, nbors_cand_size);
        for (unsigned m = 0; m < nbors_cand_size; ++m) {
          const int nbor_id = nbr_ids[m];
          const float nbor_dist = dist_scratch[m];
          if (stats != nullptr) {
            stats->n_cmps++;
          }

          Neighbor nn(nbor_id, nbor_dist, true, m < n_member);
          if (nn >= retset[cur_list_size - 1] && (cur_list_size == l_pool))
            continue;

          auto r = InsertIntoPool(retset.data(), cur_list_size, nn);
          if (cur_list_size < l_pool) {
            ++cur_list_size;
            if (unlikely(cur_list_size >= retset.size())) {
              retset.resize(2 * cur_list_size);
            }
          }
          if (r < nk)
            nk = r;
        }
      }
    };

    auto add_to_retset = [&](const unsigned *node_ids, const uint64_t n_ids, float *dists) {
      for (uint64_t i = 0; i < n_ids; ++i) {
        retset[cur_list_size++] = Neighbor(node_ids[i], dists[i], true);
        visited.insert(node_ids[i]);
      }
    };

    // --- Stats init ---
    if (stats != nullptr) {
      stats->io_us = 0;
      stats->io_us1 = 0;
      stats->cpu_us = 0;
      stats->cpu_us1 = 0;
      stats->cpu_us2 = 0;
    }

    // --- In-memory index initialization ---
    int64_t cur_beam_width = std::min<uint64_t>(4, beam_width);
    std::vector<unsigned> mem_tags(mem_L);
    std::vector<float> mem_dists(mem_L);

    if (mem_L) {
      mem_index_->search_with_tags_fast(query, mem_L, mem_tags.data(), mem_dists.data());
      add_to_retset(mem_tags.data(), std::min<uint64_t>(mem_L, l_pool), mem_dists.data());
    } else {
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, &meta_.entry_point_id, 1);
      retset[cur_list_size++] = Neighbor(meta_.entry_point_id, dist_scratch[0], true, false);
      visited.insert(meta_.entry_point_id);
    }

    // --- IO pipeline ---
    struct io_t {
      Neighbor nbr;
      unsigned page_id;
      unsigned loc;
      IORequest *read_req;
      bool operator>(const io_t &rhs) const {
        return nbr.distance > rhs.nbr.distance;
      }
      bool operator<(const io_t &rhs) const {
        return nbr.distance < rhs.nbr.distance;
      }
      bool finished() {
        return read_req->finished;
      }
    };
    std::queue<io_t> on_flight_ios;

    auto send_read_req = [&](Neighbor &item) -> bool {
      item.flag = false;
      this->lock_idx(idx_lock_table, item.id, std::vector<uint32_t>(), true);
      const unsigned loc = id2loc(item.id), pid = loc_sector_no(loc);

      uint64_t &cur_buf_idx = query_buf->sector_idx;
      auto buf = sector_scratch + cur_buf_idx * read_io_size;
      auto &req = query_buf->reqs[cur_buf_idx];
      req = IORequest(static_cast<uint64_t>(pid) * SECTOR_LEN, read_io_size, buf, u_loc_offset(loc), meta_.max_node_len,
                      sector_scratch);
      reader->send_read(req, ctx);

      on_flight_ios.push(io_t{item, pid, loc, &req});
      cur_buf_idx = (cur_buf_idx + 1) % MAX_N_SECTOR_READS;

      if (stats != nullptr) {
        stats->n_ios += (double) read_io_size / SECTOR_LEN;
      }
      return true;
    };

    std::unordered_map<unsigned, DiskNode<T>> id_buf_map;
    auto poll_all = [&]() -> std::pair<int, int> {
      if (insert_ctx != nullptr) {
        reader->poll_alloc(ctx, &insert_ctx->page_ref);
      } else {
        reader->poll(ctx);
      }
      unsigned n_in = 0, n_out = 0;
      while (!on_flight_ios.empty() && on_flight_ios.front().finished()) {
        io_t &io = on_flight_ios.front();
        id_buf_map.insert(std::make_pair(io.nbr.id, node_from_page((char *) io.read_req->buf, io.loc)));
        if (insert_ctx != nullptr) {
          insert_ctx->hint_pages.push_back(io.page_id);
        }
        io.nbr.distance <= retset[cur_list_size - 1].distance ? ++n_in : ++n_out;
        this->unlock_idx(idx_lock_table, io.nbr.id);
        on_flight_ios.pop();
      }
      return std::make_pair(n_in, n_out);
    };

    auto send_best_read_req = [&](uint32_t n) -> bool {
      unsigned n_sent = 0, marker = 0;
      while (marker < cur_list_size && n_sent < n) {
        while (marker < cur_list_size &&
               (retset[marker].flag == false || id_buf_map.find(retset[marker].id) != id_buf_map.end())) {
          retset[marker].flag = false;
          ++marker;
        }
        if (marker >= cur_list_size) {
          break;
        }
        n_sent += send_read_req(retset[marker]);
      }
      return n_sent != 0;
    };

    // --- Node selection ---
    auto calc_best_node = [&]() -> int {
      unsigned marker = 0, nk = cur_list_size, first_unvisited_eager = cur_list_size;
      for (marker = 0; marker < cur_list_size; ++marker) {
        if (!retset[marker].visited && id_buf_map.find(retset[marker].id) != id_buf_map.end()) {
          retset[marker].flag = false;
          retset[marker].visited = true;
          auto it = id_buf_map.find(retset[marker].id);
          auto &[id, node] = *it;
          if (compute_exact_dists_and_push(node, id)) {
            retset[marker].is_member = true;
          }
          compute_and_push_nbrs(node, nk);
          break;
        }
      }

      for (unsigned i = 0; i < cur_list_size; ++i) {
        if (!retset[i].visited && retset[i].flag && id_buf_map.find(retset[i].id) == id_buf_map.end()) {
          first_unvisited_eager = i;
          break;
        }
      }
      return first_unvisited_eager;
    };

    // --- Termination: N valid members OR all visited OR range boundary crossed ---
    // Range early-stop is driven by the approximate retset distance. We expand
    // the exact threshold with a fixed slack factor so only sufficiently large
    // approximate distances trigger range_crossed. Defaults to +inf
    // (disabled) so non-range callers keep original behavior.
    constexpr float kRangeEarlyStopFactor = 2.0f;
    const float approx_range_partial = range_partial * kRangeEarlyStopFactor;
    auto terminate = [&]() -> bool {
      int ret = -1;
      uint64_t is_member_cnt = 0;
      bool range_crossed = false;
      for (unsigned i = 0; i < cur_list_size; ++i) {
        is_member_cnt += retset[i].is_member;
        if (!retset[i].visited) {
          ret = i;
          break;
        }
        if (retset[i].distance > approx_range_partial) {
          range_crossed = true;
          break;
        }
      }
      return is_member_cnt >= l_search || ret == -1 || range_crossed;
    };

    // --- Main search loop ---
    auto cpu2_st = std::chrono::high_resolution_clock::now();
    send_best_read_req(cur_beam_width - on_flight_ios.size());
    unsigned marker = 0, max_marker = 0;

    if (mem_L) {  // overlap init with the first read.
      nbr_handler->initialize_query(query, query_buf);
      nbr_handler->compute_dists(query_buf, mem_tags.data(), mem_L);
      for (unsigned i = 0; i < cur_list_size; ++i) {
        retset[i].distance = dist_scratch[i];
      }
      std::sort(retset.begin(), retset.begin() + cur_list_size);
    }

    int cur_n_in = 0, cur_tot = 0;
    while (!terminate()) {
      auto [n_in, n_out] = poll_all();
      std::ignore = n_in;
      std::ignore = n_out;

      if (max_marker >= 5 && n_in + n_out > 0) {
        cur_n_in += n_in;
        cur_tot += n_in + n_out;
        constexpr double kWasteThreshold = 0.1;
        if ((cur_tot - cur_n_in) * 1.0 / cur_tot <= kWasteThreshold) {
          cur_beam_width = cur_beam_width + 1;
          cur_beam_width = std::max(cur_beam_width, 4l);
          cur_beam_width = std::min((int64_t) beam_width, cur_beam_width);
        }
      }

      if ((int64_t) on_flight_ios.size() < cur_beam_width) {
        send_best_read_req(1);
      }
      marker = calc_best_node();
      max_marker = std::max(max_marker, marker);
    }

    // Drain remaining in-flight IOs.
    while (!on_flight_ios.empty()) {
      poll_all();
    }
    calc_best_node();

    auto cpu2_ed = std::chrono::high_resolution_clock::now();
    if (stats != nullptr) {
      stats->cpu_us2 = std::chrono::duration_cast<std::chrono::microseconds>(cpu2_ed - cpu2_st).count();
      stats->cpu_us = n_computes;
    }

    std::sort(full_retset.begin(), full_retset.end(),
              [](const Neighbor &left, const Neighbor &right) { return left < right; });

    push_query_buf(query_buf);

    if (stats != nullptr) {
      stats->total_us = (double) query_timer.elapsed();
    }
  }

}  // namespace pipeann
