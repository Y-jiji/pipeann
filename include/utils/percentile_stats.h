#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "utils.h"

namespace pipeann {
  struct QueryStats {
    double total_us = 0;        // total time to process query in micros
    double n_ios = 0;           // total # of IOs issued
    double io_us = 0;           // total time spent in IO
    double io_us1 = 0;          // total time spent in IO
    double head_us = 0;         // total time spent in in-memory index
    double cpu_us = 0;          // total time spent in CPU
    double cpu_us1 = 0;         // total time spent in CPU
    double cpu_us2 = 0;         // total time spent in CPU
    double n_cmps = 0;          // # cmps
    double n_exact = 0;         // # full-precision distances computed over fetched coords
    double n_hops = 0;          // # search hops
    double n_current_used = 0;  // # force return for latency limit
    double n_rmw_reads = 0;     // # pages this insert's graph-mutation read-modify-write asked to re-read
    double n_pages_touched = 0; // # distinct disk pages touched by this insert's graph-mutation read-modify-write
    double n_out_edges = 0;     // # outgoing edges written for the newly-inserted node
    double n_in_edges = 0;      // # existing neighbors whose edge list gained a back-edge to the new node
    double n_evictions = 0;     // # of those back-edge updates that evicted a pre-existing edge
    double n_filter[N_FILTER_TYPES] = {0};
    double n_est_filter_reads[N_FILTER_TYPES] = {0};
    double n_est_filter_cmps[N_FILTER_TYPES] = {0};
    double n_filter_reads[N_FILTER_TYPES] = {0};
    double n_filter_accessed_vectors[N_FILTER_TYPES] = {0};
    double n_filter_false_positives[N_FILTER_TYPES] = {0};

    double filter_io_us[N_FILTER_TYPES] = {0};
    double filter_cpu_us[N_FILTER_TYPES] = {0};
    double filter_io_us1[N_FILTER_TYPES] = {0};
  };

  inline double get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                                     const std::function<double(const QueryStats &)> &member_fn) {
    std::vector<double> vals(len);
    for (uint64_t i = 0; i < len; i++) {
      vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const double &left, const double &right) { return left < right; });

    auto retval = vals[(uint64_t) (percentile * ((float) len))];
    vals.clear();
    return retval;
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t len,
                               const std::function<double(const QueryStats &)> &member_fn) {
    double avg = 0;
    for (uint64_t i = 0; i < len; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) len);
  }

  inline double get_mean_stats(QueryStats *stats, uint64_t cnt, uint64_t tot,
                               const std::function<double(const QueryStats &)> &member_fn) {
    if (cnt == 0) {
      return 0;
    }

    double avg = 0;
    for (uint64_t i = 0; i < tot; i++) {
      avg += member_fn(stats[i]);
    }
    return avg / ((double) cnt);
  }
}  // namespace pipeann
