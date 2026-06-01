// Insert-only growing-from-empty workload for OdinANN.
//
// Mirrors fnct-bench's odindb experiment (see fnct-bench/src/odindb.rs
// and fnct-bench/src/bench_loop.rs): start with an empty index, insert
// `batch` vectors at a time, then run one search pass per
// (k, L) configuration before the next insert batch.
//
// Data files are RAW (no .bin header), matching fnct-bench's
// load_raw/load_batch — the user supplies dim and total length on the
// command line. Result files use fnct-bench's text format so that
// fnct-bench/script/recall.py can compare them directly against the
// brute-force ground truth.

#include "dynamic_index.h"
#include "utils/timer.h"
#include "utils.h"

#include <omp.h>
#include <unistd.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using TagT = uint32_t;

struct SearchCfg {
  uint32_t k;
  uint32_t L;
};

std::string sanitize(const std::string &s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
  }
  return out;
}

// Read `count` points of `dim` floats starting at point offset `off`
// from a RAW (no header) file. Matches fnct-bench's load_batch.
template<typename T>
void load_raw_batch(const std::string &path, size_t off, size_t count, size_t dim, std::vector<T> &out) {
  out.assign(count * dim, T{});
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    LOG(ERROR) << "open data file failed: " << path;
    std::exit(1);
  }
  f.seekg(static_cast<std::streamoff>(off * dim * sizeof(T)), std::ios::beg);
  f.read(reinterpret_cast<char *>(out.data()), count * dim * sizeof(T));
  if (!f) {
    LOG(ERROR) << "short read at off=" << off << " count=" << count << " path=" << path;
    std::exit(1);
  }
}

template<typename T>
void load_raw_all(const std::string &path, size_t count, size_t dim, std::vector<T> &out) {
  load_raw_batch<T>(path, 0, count, dim, out);
}

struct SearchResult {
  std::vector<TagT> ids;     // size = qlen * topk
  std::vector<float> dists;  // size = qlen * topk
  double qps;
  double mean_ms, p50_ms, p95_ms, p99_ms, p999_ms;
};

template<typename T>
SearchResult run_search(DynamicIndex<T> &index, const std::vector<T> &queries, size_t qlen, size_t dim,
                        uint32_t topk, uint32_t L, int search_threads) {
  SearchResult sr;
  sr.ids.assign(qlen * topk, std::numeric_limits<TagT>::max());
  sr.dists.assign(qlen * topk, std::numeric_limits<float>::max());

  std::vector<double> lat_ms(qlen, 0.0);
  auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic) num_threads(search_threads)
  for (size_t i = 0; i < qlen; i++) {
    pipeann::Timer qt;
    index.search(queries.data() + i * dim, topk, L, sr.ids.data() + i * topk, sr.dists.data() + i * topk);
    lat_ms[i] = qt.elapsed() / 1000.0;  // us -> ms
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double wall = std::chrono::duration<double>(t1 - t0).count();

  sr.qps = qlen / wall;
  sr.mean_ms = std::accumulate(lat_ms.begin(), lat_ms.end(), 0.0) / static_cast<double>(qlen);
  std::sort(lat_ms.begin(), lat_ms.end());
  auto at = [&](double p) { return lat_ms[std::min<size_t>(static_cast<size_t>(p * qlen), qlen - 1)]; };
  sr.p50_ms = at(0.50);
  sr.p95_ms = at(0.95);
  sr.p99_ms = at(0.99);
  sr.p999_ms = at(0.999);
  return sr;
}

void write_batch_header(std::ofstream &of, size_t bidx, size_t offset) {
  of << "# after batch " << bidx << " (offset " << offset << ")\n";
}

void write_batch_results(std::ofstream &of, const SearchResult &sr, size_t qlen, uint32_t topk) {
  for (size_t qi = 0; qi < qlen; qi++) {
    of << "query " << qi << ":";
    for (uint32_t j = 0; j < topk; j++) {
      TagT id = sr.ids[qi * topk + j];
      float d = sr.dists[qi * topk + j];
      of << " (" << j << ": id=" << id << ", dist=" << d << ")";
    }
    of << "\n";
  }
  of.flush();
}

double rss_mb() {
  long resident = 0;
  std::ifstream f("/proc/self/statm");
  long sz = 0, sh = 0;
  if (f) f >> sz >> resident >> sh;
  long page_kb = sysconf(_SC_PAGE_SIZE) / 1024;
  return static_cast<double>(resident * page_kb) / 1024.0;
}

template<typename T>
void run_experiment(const std::string &data_path, size_t total_len, size_t batch, size_t dim,
                    const std::string &query_path, size_t qlen, const std::string &index_prefix,
                    const std::string &out_prefix, uint32_t L_insert, uint32_t max_degree, float alpha,
                    int insert_threads, int search_threads, const std::vector<SearchCfg> &search_cfgs) {
  std::vector<T> queries;
  load_raw_all<T>(query_path, qlen, dim, queries);
  LOG(INFO) << "Loaded " << qlen << " queries, dim=" << dim;

  pipeann::IndexBuildParameters params;
  params.R = max_degree;
  params.L = L_insert;
  params.alpha = alpha;
  params.num_threads = insert_threads;
  params.max_nthreads = static_cast<uint32_t>(insert_threads + search_threads);
  params.beam_width = 8;

  DynamicIndex<T> index(static_cast<uint32_t>(dim), pipeann::Metric::L2, &params);
  index.set_index_prefix(index_prefix);
  index.omp_set_num_threads(static_cast<uint32_t>(insert_threads));

  std::ofstream stats(out_prefix + "stats.tsv");
  stats << "batch\tphase\ttag\tnum_points\tpoints_inserted\t"
           "insert_throughput_pts_sec\tinsert_time_ms\tbatch_total_ms\t"
           "search_qps\tmean_lat_ms\tp50_lat_ms\tp95_lat_ms\tp99_lat_ms\tp999_lat_ms\t"
           "rss_mb\n";

  std::vector<std::unique_ptr<std::ofstream>> result_files;
  result_files.reserve(search_cfgs.size());
  for (const auto &sc : search_cfgs) {
    std::ostringstream tag;
    tag << "beamest(k=" << sc.k << ",beam=" << sc.L << ",starts=1)";
    std::string fname = out_prefix + sanitize(tag.str()) + ".txt";
    result_files.emplace_back(std::make_unique<std::ofstream>(fname));
    if (!*result_files.back()) {
      LOG(ERROR) << "cannot open result file: " << fname;
      std::exit(1);
    }
  }

  pipeann::Timer global_t;
  std::vector<T> data_load;
  size_t off = 0, bidx = 0;
  while (off < total_len) {
    size_t bsz = std::min(batch, total_len - off);
    LOG(INFO) << "[batch " << bidx << "] loading off=" << off << " bsz=" << bsz;
    load_raw_batch<T>(data_path, off, bsz, dim, data_load);

    std::vector<TagT> tags(bsz);
    for (size_t i = 0; i < bsz; i++) tags[i] = static_cast<TagT>(off + i);

    pipeann::Timer ins_t;
    index.add(data_load.data(), tags.data(), static_cast<uint32_t>(bsz));
    double ins_ms = ins_t.elapsed() / 1000.0;
    double ips = (bsz * 1e3) / ins_ms;
    LOG(INFO) << "[batch " << bidx << "] inserted " << bsz << " in " << ins_ms << "ms (" << ips << "/s)";

    pipeann::Timer batch_total_t;
    index.omp_set_num_threads(static_cast<uint32_t>(search_threads));
    for (size_t si = 0; si < search_cfgs.size(); si++) {
      const auto &sc = search_cfgs[si];
      LOG(INFO) << "[batch " << bidx << "] search k=" << sc.k << " L=" << sc.L;
      SearchResult sr = run_search<T>(index, queries, qlen, dim, sc.k, sc.L, search_threads);

      auto &of = *result_files[si];
      write_batch_header(of, bidx, off + bsz);
      write_batch_results(of, sr, qlen, sc.k);

      std::ostringstream tag;
      tag << "beamest(k=" << sc.k << ",beam=" << sc.L << ",starts=1)_run0";
      stats << bidx << "\tsearch\t" << tag.str() << "\t\t\t\t\t\t"
            << std::fixed << std::setprecision(2)
            << sr.qps << "\t" << sr.mean_ms << "\t" << sr.p50_ms << "\t"
            << sr.p95_ms << "\t" << sr.p99_ms << "\t" << sr.p999_ms << "\t"
            << rss_mb() << "\n";
    }
    index.omp_set_num_threads(static_cast<uint32_t>(insert_threads));
    double batch_ms = batch_total_t.elapsed() / 1000.0 + ins_ms;

    stats << bidx << "\tinsert\t\t" << bsz << "\t" << (off + bsz) << "\t"
          << std::fixed << std::setprecision(2)
          << ips << "\t" << ins_ms << "\t" << batch_ms << "\t\t\t\t\t\t\t"
          << rss_mb() << "\n";
    stats.flush();

    LOG(INFO) << "[batch " << bidx << "] done, elapsed=" << (global_t.elapsed() / 1e6) << "s rss="
              << rss_mb() << "MB";
    off += bsz;
    bidx++;
  }
  LOG(INFO) << "All batches done in " << (global_t.elapsed() / 1e6) << "s";
}

void usage(const char *prog) {
  std::cerr
      << "Usage: " << prog
      << " <dtype:float|int8|uint8> <data_file_raw> <dim> <total_len> <batch>\n"
      << "            <query_file_raw> <qlen>\n"
      << "            <index_prefix> <out_prefix>\n"
      << "            <L_insert> <max_degree> <alpha>\n"
      << "            <insert_threads> <search_threads>\n"
      << "            <k1> <L1> [<k2> <L2> ...]\n"
      << "Notes:\n"
      << "  data_file_raw and query_file_raw are RAW (no .bin header).\n"
      << "  out_prefix is used literally; parent directory must exist.\n"
      << "  Result files: <out_prefix>beamest_k_<k>_beam_<L>_starts_1_.txt\n"
      << "  Stats: <out_prefix>stats.tsv\n";
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 17 || ((argc - 15) % 2) != 0) {
    usage(argv[0]);
    return 1;
  }
  int a = 1;
  std::string dtype = argv[a++];
  std::string data_path = argv[a++];
  size_t dim = std::stoul(argv[a++]);
  size_t total_len = std::stoul(argv[a++]);
  size_t batch = std::stoul(argv[a++]);
  std::string query_path = argv[a++];
  size_t qlen = std::stoul(argv[a++]);
  std::string index_prefix = argv[a++];
  std::string out_prefix = argv[a++];
  uint32_t L_insert = static_cast<uint32_t>(std::stoul(argv[a++]));
  uint32_t max_degree = static_cast<uint32_t>(std::stoul(argv[a++]));
  float alpha = std::stof(argv[a++]);
  int insert_threads = std::stoi(argv[a++]);
  int search_threads = std::stoi(argv[a++]);

  std::vector<SearchCfg> search_cfgs;
  for (; a + 1 < argc; a += 2) {
    SearchCfg sc;
    sc.k = static_cast<uint32_t>(std::stoul(argv[a]));
    sc.L = static_cast<uint32_t>(std::stoul(argv[a + 1]));
    search_cfgs.push_back(sc);
  }

  LOG(INFO) << "dtype=" << dtype << " dim=" << dim << " total_len=" << total_len << " batch=" << batch;
  LOG(INFO) << "qlen=" << qlen << " L_insert=" << L_insert << " R=" << max_degree << " alpha=" << alpha;
  LOG(INFO) << "insert_threads=" << insert_threads << " search_threads=" << search_threads;
  for (const auto &sc : search_cfgs) {
    LOG(INFO) << "search cfg k=" << sc.k << " L=" << sc.L;
  }

  if (dtype == "float") {
    run_experiment<float>(data_path, total_len, batch, dim, query_path, qlen, index_prefix, out_prefix, L_insert,
                          max_degree, alpha, insert_threads, search_threads, search_cfgs);
  } else if (dtype == "int8") {
    run_experiment<int8_t>(data_path, total_len, batch, dim, query_path, qlen, index_prefix, out_prefix, L_insert,
                           max_degree, alpha, insert_threads, search_threads, search_cfgs);
  } else if (dtype == "uint8") {
    run_experiment<uint8_t>(data_path, total_len, batch, dim, query_path, qlen, index_prefix, out_prefix, L_insert,
                            max_degree, alpha, insert_threads, search_threads, search_cfgs);
  } else {
    LOG(ERROR) << "unsupported dtype: " << dtype;
    return 1;
  }
  return 0;
}
