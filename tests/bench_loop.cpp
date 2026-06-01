// Minimal port of fnct-bench/src/bench_loop.rs into the PipeANN tree.
//
// Redacted vs. fnct-bench:
//   - no visitor config: PipeANN's search path is fixed (pipe_search,
//     beam_width=32 hardcoded). The only knob we expose is L (search list)
//     and k (top-k).
//   - no engine config: PipeANN is *the* engine.
//   - no runtime config: PipeANN is vendor-locked to liburing.
//
// What is left is the [dataset] config (dim, dtype, data, queries, len,
// qlen, batch) plus the bench_loop behaviour: read `batch` raw vectors,
// bulk-insert them with tag = global offset, then run each configured
// (k, L) sweep across all queries and emit one CSV row per query in the
// exact QueryRow column order produced by bench_loop.rs:
//
//   batch, visitor, qi, start_ns, lat_ns,
//   id@1, dist@1, ..., id@K, dist@K,
//   n_ios, n_hops, n_cmps, total_us
//
// `visitor` is the literal string "pipeann(k=K,L=L)_run0" so existing
// fnct-bench scripts that key on the visitor column keep working.
//
// Data and query files are RAW (no .bin header), matching fnct-bench's
// load_raw / load_batch.

#include "dynamic_index.h"
#include "utils/percentile_stats.h"
#include "utils/timer.h"
#include "utils.h"

#include <omp.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using TagT = uint32_t;
using clk = std::chrono::steady_clock;

struct SearchCfg {
  uint32_t k;
  uint32_t L;
};

struct DataCfg {
  std::string dtype;     // "float" | "int8" | "uint8"
  std::string data;      // raw flat file, no header
  std::string queries;   // raw flat file, no header
  size_t dim;
  size_t len;
  size_t qlen;
  size_t batch;
};

template<typename T>
void load_raw_batch(const std::string &path, size_t off, size_t count, size_t dim, std::vector<T> &out) {
  out.assign(count * dim, T{});
  std::ifstream f(path, std::ios::binary);
  if (!f) { LOG(ERROR) << "open " << path; std::exit(1); }
  f.seekg(static_cast<std::streamoff>(off * dim * sizeof(T)), std::ios::beg);
  f.read(reinterpret_cast<char *>(out.data()), count * dim * sizeof(T));
  if (!f) { LOG(ERROR) << "short read " << path; std::exit(1); }
}

// Write the QueryRow CSV header once, matching the field order the Rust
// QueryRow::serialize impl emits (5 fixed cols, then id@N/dist@N pairs,
// then DBStats columns).
void write_header(std::ofstream &of, uint32_t max_k) {
  of << "batch,visitor,qi,start_ns,lat_ns";
  for (uint32_t i = 1; i <= max_k; i++) of << ",id@" << i << ",dist@" << i;
  of << ",n_ios,n_hops,n_cmps,total_us\n";
}

template<typename T>
void run_search_sweep(std::ofstream &of, DynamicIndex<T> &index, const std::vector<T> &queries,
                      size_t qlen, size_t dim, const SearchCfg &sc, uint32_t max_k, size_t bidx,
                      clk::time_point run_start, int search_threads) {
  std::vector<TagT> ids(qlen * sc.k, std::numeric_limits<TagT>::max());
  std::vector<float> dists(qlen * sc.k, std::numeric_limits<float>::max());
  std::vector<pipeann::QueryStats> stats(qlen);
  std::vector<uint64_t> start_ns(qlen, 0);
  std::vector<uint64_t> lat_ns(qlen, 0);

#pragma omp parallel for schedule(dynamic) num_threads(search_threads)
  for (size_t qi = 0; qi < qlen; qi++) {
    auto t = clk::now();
    start_ns[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
    index.search(queries.data() + qi * dim, sc.k, sc.L,
                 ids.data() + qi * sc.k, dists.data() + qi * sc.k, &stats[qi]);
    lat_ns[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
  }

  std::ostringstream vis;
  vis << "pipeann(k=" << sc.k << ",L=" << sc.L << ")_run0";
  for (size_t qi = 0; qi < qlen; qi++) {
    of << bidx << ',' << vis.str() << ',' << qi << ',' << start_ns[qi] << ',' << lat_ns[qi];
    for (uint32_t j = 0; j < max_k; j++) {
      if (j < sc.k) {
        TagT id = ids[qi * sc.k + j];
        float d = dists[qi * sc.k + j];
        if (id == std::numeric_limits<TagT>::max()) of << ",,";
        else of << ',' << id << ',' << d;
      } else {
        of << ",,";
      }
    }
    const auto &s = stats[qi];
    of << ',' << static_cast<uint64_t>(s.n_ios)
       << ',' << static_cast<uint64_t>(s.n_hops)
       << ',' << static_cast<uint64_t>(s.n_cmps)
       << ',' << static_cast<uint64_t>(s.total_us)
       << '\n';
  }
  of.flush();
}

template<typename T>
void run(const DataCfg &ds, const std::string &index_prefix, const std::string &out_prefix,
         uint32_t L_insert, uint32_t R, float alpha, int insert_threads, int search_threads,
         const std::vector<SearchCfg> &cfgs) {
  std::vector<T> queries;
  load_raw_batch<T>(ds.queries, 0, ds.qlen, ds.dim, queries);

  pipeann::IndexBuildParameters params;
  params.R = R;
  params.L = L_insert;
  params.alpha = alpha;
  params.num_threads = insert_threads;
  params.max_nthreads = static_cast<uint32_t>(insert_threads + search_threads);
  params.beam_width = 8;

  DynamicIndex<T> index(static_cast<uint32_t>(ds.dim), pipeann::Metric::L2, &params);
  index.set_index_prefix(index_prefix);

  uint32_t max_k = 0;
  for (const auto &sc : cfgs) max_k = std::max(max_k, sc.k);

  std::ofstream of(out_prefix + "records.csv");
  if (!of) { LOG(ERROR) << "open " << out_prefix << "records.csv"; std::exit(1); }
  write_header(of, max_k);

  auto run_start = clk::now();
  std::vector<T> batch_buf;
  size_t off = 0, bidx = 0;
  while (off < ds.len) {
    size_t bsz = std::min(ds.batch, ds.len - off);
    load_raw_batch<T>(ds.data, off, bsz, ds.dim, batch_buf);
    std::vector<TagT> tags(bsz);
    for (size_t i = 0; i < bsz; i++) tags[i] = static_cast<TagT>(off + i);

    index.omp_set_num_threads(static_cast<uint32_t>(insert_threads));
    pipeann::Timer it;
    index.add(batch_buf.data(), tags.data(), static_cast<uint32_t>(bsz));
    double ins_ms = it.elapsed() / 1000.0;
    LOG(INFO) << "[batch " << bidx << "] inserted " << bsz << " in " << ins_ms << "ms";

    index.omp_set_num_threads(static_cast<uint32_t>(search_threads));
    for (const auto &sc : cfgs) {
      LOG(INFO) << "[batch " << bidx << "] search k=" << sc.k << " L=" << sc.L;
      run_search_sweep<T>(of, index, queries, ds.qlen, ds.dim, sc, max_k, bidx, run_start, search_threads);
    }
    off += bsz;
    bidx++;
  }
}

void usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " <dtype:float|int8|uint8> <data_raw> <dim> <len> <batch>\n"
      << "       <queries_raw> <qlen>\n"
      << "       <index_prefix> <out_prefix>\n"
      << "       <L_insert> <R> <alpha> <insert_threads> <search_threads>\n"
      << "       <k1> <L1> [<k2> <L2> ...]\n"
      << "Emits <out_prefix>records.csv in bench_loop.rs QueryRow format.\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 17 || ((argc - 15) % 2) != 0) { usage(argv[0]); return 1; }
  int a = 1;
  DataCfg ds;
  ds.dtype = argv[a++];
  ds.data = argv[a++];
  ds.dim = std::stoul(argv[a++]);
  ds.len = std::stoul(argv[a++]);
  ds.batch = std::stoul(argv[a++]);
  ds.queries = argv[a++];
  ds.qlen = std::stoul(argv[a++]);
  std::string index_prefix = argv[a++];
  std::string out_prefix = argv[a++];
  uint32_t L_insert = static_cast<uint32_t>(std::stoul(argv[a++]));
  uint32_t R = static_cast<uint32_t>(std::stoul(argv[a++]));
  float alpha = std::stof(argv[a++]);
  int insert_threads = std::stoi(argv[a++]);
  int search_threads = std::stoi(argv[a++]);

  std::vector<SearchCfg> cfgs;
  for (; a + 1 < argc; a += 2) {
    cfgs.push_back({static_cast<uint32_t>(std::stoul(argv[a])),
                    static_cast<uint32_t>(std::stoul(argv[a + 1]))});
  }

  if (ds.dtype == "float") {
    run<float>(ds, index_prefix, out_prefix, L_insert, R, alpha, insert_threads, search_threads, cfgs);
  } else if (ds.dtype == "int8") {
    run<int8_t>(ds, index_prefix, out_prefix, L_insert, R, alpha, insert_threads, search_threads, cfgs);
  } else if (ds.dtype == "uint8") {
    run<uint8_t>(ds, index_prefix, out_prefix, L_insert, R, alpha, insert_threads, search_threads, cfgs);
  } else {
    LOG(ERROR) << "unsupported dtype: " << ds.dtype; return 1;
  }
  return 0;
}
