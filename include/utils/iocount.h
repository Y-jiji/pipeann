#pragma once

#include <cstdint>

// Device-level read accounting. QueryStats::n_ios and n_rmw_reads count the
// pages the search and the read-modify-write asked for, which the user-space
// page cache may serve without touching the drive. This counter is bumped
// where a read is actually submitted to the device, so a caller that samples
// it before and after an operation learns what that operation cost the drive.
// It is thread-local, so an operation must sample it on the thread that runs
// it; work a background thread defers is not included.

namespace pipeann {
  extern thread_local uint64_t tls_dev_reads;

  /** @brief Pages this thread has read from the device since it started.
   *  @return monotonically growing 4 KB page count
   */
  inline uint64_t dev_reads() {
    return tls_dev_reads;
  }
}  // namespace pipeann
