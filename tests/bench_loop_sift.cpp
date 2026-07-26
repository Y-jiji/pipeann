// SIFT-200M PipeANN two-phase bench, ported from fnct-bench's Bench harness
// (see fnct-db/evaluation/hermes_sift.rs for the reference shape).
//
// Phase 1 (0 -> BASE): plain sequential insert, no search. The very first
// batch goes through DynamicIndex::add() (batch API) because that is the
// only path that flips the index from its initial in-memory mode into the
// disk-resident mode (DynamicIndex::kBuildThreshold check lives in add(),
// not in the single-point insert()); every later insert -- including all of
// phase 2 -- goes through insert() per point so a QueryStats can be threaded
// through (see dynamic_index.h / ssd_index.h insert_in_place changes),
// giving real per-insert n_ios/n_cmps/total_us instead of only a wall-clock
// batch latency.
//
// Phase 2 (BASE -> N): each BATCH-point step runs ONE concurrent insert+
// search step: a single shuffled index range over [insert-batch | fixed
// query set] is dispatched across one OpenMP thread pool, exactly mirroring
// fnct-bench's Bench::runmix (see fnct-bench/src/lib.rs). Rows from this
// step alone are undersampled (keep 1-in-MIXRATE). It is followed by ONE
// fully-logged (rate=1) search-only call using the same PRIMARY_CFG -- the
// recall-verification call, since the mix step's own search rows are too
// sparse (undersampled) to compute recall@10 from reliably. Recall is
// computed post-hoc against a ground-truth log, so every logged row still
// carries the returned ids/dists needed for that join.
//
// Data/queries are raw flat files (no header), dtype uint8_t, dim 128.

#include "dynamic_index.h"
#include "utils/percentile_stats.h"
#include "utils/timer.h"
#include "utils.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using T = uint8_t;
using TagT = uint32_t;
using clk = std::chrono::steady_clock;

constexpr size_t DIM = 128;
constexpr size_t N = 200'000'000;
constexpr size_t BASE = 100'000'000;
constexpr size_t BATCH = 1'000'000;
constexpr size_t QLEN = 10'000;

constexpr uint32_t L_INSERT = 128;
constexpr uint32_t R = 64;
constexpr float ALPHA = 1.2f;
constexpr int INSERT_THREADS = 16;
constexpr int SEARCH_THREADS = 16;
constexpr int MIX_THREADS = 32;

// Keep 1-in-MIXRATE rows from the concurrent mix step only; phase 1 and the
// recall-verification search call are always logged in full.
constexpr uint64_t MIXRATE = 50;

struct SearchCfg {
  uint32_t k;
  uint32_t L;
};

// The one config used both by the mix step and the recall-verification
// search call that follows it -- same config, so throughput/latency and
// recall@10 are measured on identical search behaviour.
constexpr SearchCfg PRIMARY_CFG{10, 120};

void load_raw_batch(const std::string &path, size_t off, size_t count, size_t dim, std::vector<T> &out) {
  out.assign(count * dim, T{});
  std::ifstream f(path, std::ios::binary);
  if (!f) { LOG(ERROR) << "open " << path; std::exit(1); }
  f.seekg(static_cast<std::streamoff>(off * dim * sizeof(T)), std::ios::beg);
  f.read(reinterpret_cast<char *>(out.data()), count * dim * sizeof(T));
  if (!f) { LOG(ERROR) << "short read " << path; std::exit(1); }
}

// Column order matches fnct-bench's QueryRow, plus a leading `op` column
// distinguishing insert rows (blank id@/dist@) from search rows. `pages`,
// `out-edges`, `in-edges`, `evictions` are insert-only graph-mutation
// counters (0 on search rows) -- `pages` is the distinct-disk-page
// footprint of the insert's read-modify-write, the key page-touch metric.
void write_header(std::ofstream &of, uint32_t max_k) {
  of << "batch,op,visitor,qi,start_ns,lat_ns";
  for (uint32_t i = 1; i <= max_k; i++) of << ",id@" << i << ",dist@" << i;
  of << ",loads,pages,hops,visits,total_us,out-edges,in-edges,evictions\n";
}

void write_insert_row(std::ofstream &of, size_t bidx, const std::string &visitor, size_t qi, uint64_t start_ns,
                      uint64_t lat_ns, const pipeann::QueryStats &s, uint32_t max_k) {
  of << bidx << ",insert," << visitor << ',' << qi << ',' << start_ns << ',' << lat_ns;
  for (uint32_t j = 0; j < max_k; j++) of << ",,";
  of << ',' << static_cast<uint64_t>(s.n_ios) << ',' << static_cast<uint64_t>(s.n_pages_touched) << ','
     << static_cast<uint64_t>(s.n_hops) << ',' << static_cast<uint64_t>(s.n_cmps) << ','
     << static_cast<uint64_t>(s.total_us) << ',' << static_cast<uint64_t>(s.n_out_edges) << ','
     << static_cast<uint64_t>(s.n_in_edges) << ',' << static_cast<uint64_t>(s.n_evictions) << '\n';
}

void write_search_row(std::ofstream &of, size_t bidx, const std::string &visitor, size_t qi, uint64_t start_ns,
                      uint64_t lat_ns, const TagT *ids, const float *dists, uint32_t k, uint32_t max_k,
                      const pipeann::QueryStats &s) {
  of << bidx << ",search," << visitor << ',' << qi << ',' << start_ns << ',' << lat_ns;
  for (uint32_t j = 0; j < max_k; j++) {
    if (j < k && ids[j] != std::numeric_limits<TagT>::max()) of << ',' << ids[j] << ',' << dists[j];
    else of << ",,";
  }
  of << ',' << static_cast<uint64_t>(s.n_ios) << ',' << static_cast<uint64_t>(s.n_pages_touched) << ','
     << static_cast<uint64_t>(s.n_hops) << ',' << static_cast<uint64_t>(s.n_cmps) << ','
     << static_cast<uint64_t>(s.total_us) << ',' << static_cast<uint64_t>(s.n_out_edges) << ','
     << static_cast<uint64_t>(s.n_in_edges) << ',' << static_cast<uint64_t>(s.n_evictions) << '\n';
}

// Fully-logged search-only sweep over the fixed query set at the corpus
// size just reached, for the throughput/recall analysis. Mirrors the
// original bench_loop.cpp's run_search_sweep.
void run_search_sweep(std::ofstream &of, DynamicIndex<T> &index, const std::vector<T> &queries, const SearchCfg &sc,
                      uint32_t max_k, size_t bidx, clk::time_point run_start, const std::string &suffix) {
  std::vector<TagT> ids(QLEN * sc.k, std::numeric_limits<TagT>::max());
  std::vector<float> dists(QLEN * sc.k, std::numeric_limits<float>::max());
  std::vector<pipeann::QueryStats> stats(QLEN);
  std::vector<uint64_t> start_ns(QLEN, 0), lat_ns(QLEN, 0);

  index.omp_set_num_threads(SEARCH_THREADS);
#pragma omp parallel for schedule(dynamic) num_threads(SEARCH_THREADS)
  for (size_t qi = 0; qi < QLEN; qi++) {
    auto t = clk::now();
    start_ns[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
    index.search(queries.data() + qi * DIM, sc.k, sc.L, ids.data() + qi * sc.k, dists.data() + qi * sc.k, &stats[qi]);
    lat_ns[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
  }

  std::ostringstream vis;
  vis << "pipeann(k=" << sc.k << ",L=" << sc.L << ")" << suffix;
  for (size_t qi = 0; qi < QLEN; qi++) {
    write_search_row(of, bidx, vis.str(), qi, start_ns[qi], lat_ns[qi], ids.data() + qi * sc.k,
                     dists.data() + qi * sc.k, sc.k, max_k, stats[qi]);
  }
  of.flush();
}

// Phase 1: fully-logged sequential insert of one batch. `use_add` selects
// the once-only DynamicIndex::add() path that can trigger the mem->disk
// transform; every other call uses per-point insert() so a QueryStats can
// be captured.
void run_insert_batch(std::ofstream &of, DynamicIndex<T> &index, const std::vector<T> &batch, size_t off, size_t bsz,
                      size_t bidx, clk::time_point run_start, uint32_t max_k, bool use_add) {
  if (use_add) {
    std::vector<TagT> tags(bsz);
    for (size_t i = 0; i < bsz; i++) tags[i] = static_cast<TagT>(off + i);
    index.omp_set_num_threads(INSERT_THREADS);
    auto t = clk::now();
    index.add(batch.data(), tags.data(), static_cast<uint32_t>(bsz));
    uint64_t lat_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
    uint64_t start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
    write_insert_row(of, bidx, "pipeann_insert_add(L=" + std::to_string(L_INSERT) + ",R=" + std::to_string(R) + ")",
                     0, start_ns, lat_ns, pipeann::QueryStats{}, max_k);
    of.flush();
    return;
  }

  std::vector<pipeann::QueryStats> stats(bsz);
  std::vector<uint64_t> start_ns(bsz, 0), lat_ns(bsz, 0);
#pragma omp parallel for schedule(dynamic) num_threads(INSERT_THREADS)
  for (size_t i = 0; i < bsz; i++) {
    auto t = clk::now();
    start_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
    index.insert(batch.data() + i * DIM, static_cast<TagT>(off + i), nullptr, &stats[i]);
    lat_ns[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
  }
  std::string visitor = "pipeann_insert(L=" + std::to_string(L_INSERT) + ",R=" + std::to_string(R) + ")";
  for (size_t i = 0; i < bsz; i++) write_insert_row(of, bidx, visitor, i, start_ns[i], lat_ns[i], stats[i], max_k);
  of.flush();
}

// Phase-2 windowed step: one BATCH-point insert running concurrently with
// one search-config pass over the fixed query set. A single shuffled index
// range over [0, bsz+QLEN) is dispatched across one OpenMP thread pool --
// index < bsz is an insert job, index >= bsz is a search job -- so insert
// and search truly interleave, mirroring fnct-bench's Bench::runmix. Rows
// are undersampled post-hoc (keep 1-in-MIXRATE, in shuffled order).
void run_mix_step(std::ofstream &of, DynamicIndex<T> &index, const std::vector<T> &batch,
                  const std::vector<T> &queries, size_t off, size_t bsz, size_t bidx, clk::time_point run_start,
                  uint32_t max_k) {
  std::vector<size_t> remap(bsz + QLEN);
  std::iota(remap.begin(), remap.end(), 0);
  std::shuffle(remap.begin(), remap.end(), std::mt19937(43));

  std::vector<pipeann::QueryStats> ins_stats(bsz);
  std::vector<uint64_t> ins_start(bsz, 0), ins_lat(bsz, 0);

  std::vector<TagT> s_ids(QLEN * PRIMARY_CFG.k, std::numeric_limits<TagT>::max());
  std::vector<float> s_dists(QLEN * PRIMARY_CFG.k, std::numeric_limits<float>::max());
  std::vector<pipeann::QueryStats> s_stats(QLEN);
  std::vector<uint64_t> s_start(QLEN, 0), s_lat(QLEN, 0);

  index.omp_set_num_threads(MIX_THREADS);
#pragma omp parallel for schedule(dynamic) num_threads(MIX_THREADS)
  for (size_t j = 0; j < remap.size(); j++) {
    size_t idx = remap[j];
    auto t = clk::now();
    if (idx < bsz) {
      ins_start[idx] = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
      index.insert(batch.data() + idx * DIM, static_cast<TagT>(off + idx), nullptr, &ins_stats[idx]);
      ins_lat[idx] = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
    } else {
      size_t qi = idx - bsz;
      s_start[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(t - run_start).count();
      index.search(queries.data() + qi * DIM, PRIMARY_CFG.k, PRIMARY_CFG.L, s_ids.data() + qi * PRIMARY_CFG.k,
                   s_dists.data() + qi * PRIMARY_CFG.k, &s_stats[qi]);
      s_lat[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t).count();
    }
  }

  std::string ivis = "pipeann_insert(L=" + std::to_string(L_INSERT) + ",R=" + std::to_string(R) + ")_mix";
  std::ostringstream svis;
  svis << "pipeann(k=" << PRIMARY_CFG.k << ",L=" << PRIMARY_CFG.L << ")_mix";
  for (size_t j = 0; j < remap.size(); j++) {
    if (j % MIXRATE != 0) continue;
    size_t idx = remap[j];
    if (idx < bsz) {
      write_insert_row(of, bidx, ivis, idx, ins_start[idx], ins_lat[idx], ins_stats[idx], max_k);
    } else {
      size_t qi = idx - bsz;
      write_search_row(of, bidx, svis.str(), qi, s_start[qi], s_lat[qi], s_ids.data() + qi * PRIMARY_CFG.k,
                       s_dists.data() + qi * PRIMARY_CFG.k, PRIMARY_CFG.k, max_k, s_stats[qi]);
    }
  }
  of.flush();
}

void run(const std::string &data, const std::string &queries_path, const std::string &index_prefix,
        const std::string &out_path) {
  std::vector<T> queries;
  load_raw_batch(queries_path, 0, QLEN, DIM, queries);

  pipeann::IndexBuildParameters params;
  params.R = R;
  params.L = L_INSERT;
  params.alpha = ALPHA;
  params.num_threads = INSERT_THREADS;
  params.max_nthreads = static_cast<uint32_t>(INSERT_THREADS + SEARCH_THREADS + MIX_THREADS);
  params.beam_width = 8;

  DynamicIndex<T> index(static_cast<uint32_t>(DIM), pipeann::Metric::L2, &params);
  index.set_index_prefix(index_prefix);

  uint32_t max_k = PRIMARY_CFG.k;

  std::ofstream of(out_path);
  if (!of) { LOG(ERROR) << "open " << out_path; std::exit(1); }
  write_header(of, max_k);

  auto run_start = clk::now();
  std::vector<T> batch_buf;

  size_t off = 0, bidx = 0;
  while (off < BASE) {
    size_t bsz = std::min(BATCH, BASE - off);
    load_raw_batch(data, off, bsz, DIM, batch_buf);
    LOG(INFO) << "[phase1 batch " << bidx << "] inserting " << bsz << " at offset " << off;
    run_insert_batch(of, index, batch_buf, off, bsz, bidx, run_start, max_k, bidx == 0);
    off += bsz;
    bidx++;
  }

  while (off < N) {
    size_t bsz = std::min(BATCH, N - off);
    load_raw_batch(data, off, bsz, DIM, batch_buf);
    LOG(INFO) << "[phase2 batch " << bidx << "] mix insert+search " << bsz << " at offset " << off;
    run_mix_step(of, index, batch_buf, queries, off, bsz, bidx, run_start, max_k);
    run_search_sweep(of, index, queries, PRIMARY_CFG, max_k, bidx, run_start, "_verify");
    off += bsz;
    bidx++;
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " <data_raw> <queries_raw> <index_prefix> <out_csv>\n";
    return 1;
  }
  run(argv[1], argv[2], argv[3], argv[4]);
  return 0;
}
