#include "condaref.hpp"
#include "linux_aligned_file_reader.h"
#include "nbr/abstract_nbr.h"
#include "ssd_index.h"
#include CONDA_DISTANCE

#include <omp.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

/** @brief Rows of the synthetic dataset, the base and every inserted row */
constexpr unsigned COUNT = 3000;
/** @brief Dimension, a multiple of 8 so condaann pads nothing */
constexpr unsigned DIM = 96;
/** @brief CONDA's R and the disk record's capacity */
constexpr unsigned RANGE = 16;
/** @brief CONDA's insert L */
constexpr unsigned BUILD_L = 64;
/** @brief Rows the reference builds by inserts before the disk copy */
constexpr unsigned BASE = 2000;
/** @brief Rows deleted, then rows inserted, per round */
constexpr unsigned STEP = 200;
/** @brief Delete and insert rounds after the build */
constexpr unsigned ROUNDS = 3;
/** @brief Seed of the synthetic dataset */
constexpr unsigned SEED = 1;

/** @brief Neighbor handler whose codes are the vectors themselves, distances by condaann's own L2,
 *  the idealization the port is compared under */
template <class T>
class ExactNeighbor : public pipeann::AbstractNeighbor<T> {
 public:
  /**
   * @brief A handler over the loaded rows
   * @param metric the metric, L2
   * @param base the loaded rows, row i at id i
   * @param dim the dimension
   */
  ExactNeighbor(pipeann::Metric metric, const std::vector<T> &base, uint32_t dim);

  /**
   * @brief Bytes a query keeps in its buffer, the query itself
   * @return dim times the element size
   */
  uint64_t query_ctx_size() override;

  /**
   * @brief The handler's name
   * @return "ExactNeighbor"
   */
  std::string get_name() override;

  /**
   * @brief Keep the query in the buffer
   * @param query the query
   * @param query_buf the buffer
   */
  void initialize_query(const T *query, pipeann::QueryBuffer *query_buf) override;

  /**
   * @brief Distances from each id to the kept query, into the buffer's distance scratch
   * @param query_buf the buffer
   * @param ids the ids
   * @param n_ids how many
   */
  void compute_dists(pipeann::QueryBuffer *query_buf, const uint32_t *ids,
                     const uint64_t n_ids) override;

  /**
   * @brief Distances from one stored id to each id
   * @param query_id the stored id
   * @param ids the ids
   * @param n_ids how many
   * @param dists_out the distances
   * @param aligned_scratch unused
   */
  void compute_dists(const uint32_t query_id, const uint32_t *ids, const uint64_t n_ids,
                     float *dists_out, uint8_t *aligned_scratch) override;

  /**
   * @brief Store a vector at an id
   * @param point the vector
   * @param loc the id
   */
  void insert(T *point, uint32_t loc) override;

 private:
  std::vector<T> table;
  uint32_t width;
  std::function<float(const T *, const T *)> meter;
};

/**
 * @brief Gaussian rows, distinct in every distance with probability one
 * @param count rows
 * @param dim dimension
 * @param seed seed
 * @return count times dim floats
 */
static std::vector<float> dataset(unsigned count, unsigned dim, unsigned seed);

/**
 * @brief Write the reference's graph and rows as a base bin and a flat graph for save_from_graph
 * @param reference the reference after its insert-only build, location i holding tag i
 * @param data the rows
 * @param count locations to write
 * @param prefix path prefix of {prefix}_base.bin and {prefix}_graph.flat
 */
static void dump(const CondaRef &reference, const std::vector<float> &data, unsigned count,
                 const std::string &prefix);

/**
 * @brief Whether every location holds the same out-neighbor multiset in both, printing the first
 *  difference
 * @param reference condaann's index
 * @param port the disk port
 * @param count locations to compare
 * @return true when all match
 */
static bool same(const CondaRef &reference, pipeann::SSDIndex<float> &port, unsigned count);

/**
 * @brief Build by inserts in the reference, copy the graph to disk, then run delete and insert
 *  rounds on both serially, comparing every list after each step; SHAME(TALLFUNC)
 * @param argc argument count
 * @param argv optional working directory
 * @return 0 when every step matches
 */
int main(int argc, char **argv);

template <class T>
ExactNeighbor<T>::ExactNeighbor(pipeann::Metric metric, const std::vector<T> &base, uint32_t dim)
    : pipeann::AbstractNeighbor<T>(metric), table(base), width(dim) {
  // [1]
  efanna2e::DistanceL2<T> l2;
  meter = [l2, dim](const T *a, const T *b) { return l2.compare(a, b, dim); };
}

template <class T>
uint64_t ExactNeighbor<T>::query_ctx_size() {
  // [1]
  return width * sizeof(T);
}

template <class T>
std::string ExactNeighbor<T>::get_name() {
  // [1]
  return "ExactNeighbor";
}

template <class T>
void ExactNeighbor<T>::initialize_query(const T *query, pipeann::QueryBuffer *query_buf) {
  // [1]
  std::memcpy(query_buf->nbr_ctx_scratch, query, width * sizeof(T));
}

template <class T>
void ExactNeighbor<T>::compute_dists(pipeann::QueryBuffer *query_buf, const uint32_t *ids,
                                     const uint64_t n_ids) {
  // [1]
  const T *query = (const T *) query_buf->nbr_ctx_scratch;
  for (uint64_t i = 0; i < n_ids; ++i) {
    query_buf->aligned_dist_scratch[i] = meter(table.data() + (size_t) ids[i] * width, query);
  }
}

template <class T>
void ExactNeighbor<T>::compute_dists(const uint32_t query_id, const uint32_t *ids,
                                     const uint64_t n_ids, float *dists_out, uint8_t *aligned_scratch) {
  // [1]
  const T *query = table.data() + (size_t) query_id * width;
  for (uint64_t i = 0; i < n_ids; ++i) {
    dists_out[i] = meter(query, table.data() + (size_t) ids[i] * width);
  }
}

template <class T>
void ExactNeighbor<T>::insert(T *point, uint32_t loc) {
  // [1]
  if (table.size() < (size_t) (loc + 1) * width) {
    table.resize((size_t) (loc + 1) * width);
  }
  std::memcpy(table.data() + (size_t) loc * width, point, width * sizeof(T));
}

static std::vector<float> dataset(unsigned count, unsigned dim, unsigned seed) {
  // [1]
  std::mt19937 gen(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  std::vector<float> data((size_t) count * dim);
  for (auto &x : data) {
    x = normal(gen);
  }
  return data;
}

static void dump(const CondaRef &reference, const std::vector<float> &data, unsigned count,
                 const std::string &prefix) {
  // [1]
  std::ofstream base(prefix + "_base.bin", std::ios::binary);
  const uint32_t npts = count, dim = DIM;
  base.write((const char *) &npts, sizeof(uint32_t));
  base.write((const char *) &dim, sizeof(uint32_t));
  base.write((const char *) data.data(), (size_t) count * DIM * sizeof(float));
  // [2]
  std::ofstream graph(prefix + "_graph.flat", std::ios::binary);
  for (unsigned loc = 0; loc < count; ++loc) {
    std::vector<uint32_t> row(RANGE, pipeann::SSDIndex<float>::kGraphPad);
    std::vector<unsigned> list = reference.edges(loc);
    std::copy(list.begin(), list.end(), row.begin());
    graph.write((const char *) row.data(), RANGE * sizeof(uint32_t));
  }
}

static bool same(const CondaRef &reference, pipeann::SSDIndex<float> &port, unsigned count) {
  // [1]
  for (unsigned loc = 0; loc < count; ++loc) {
    std::vector<unsigned> want = reference.edges(loc);
    std::vector<uint32_t> got = port.condaedges(loc);
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    if (want.size() == got.size() && std::equal(want.begin(), want.end(), got.begin())) {
      continue;
    }
    // [2]
    std::printf("location %u differs: reference", loc);
    for (unsigned v : want) {
      std::printf(" %u", v);
    }
    std::printf(", port");
    for (uint32_t v : got) {
      std::printf(" %u", v);
    }
    std::printf("\n");
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  // [1]
  omp_set_num_threads(1);
  const std::string dir = argc > 1 ? argv[1] : "/tmp/test_conda";
  std::filesystem::create_directories(dir);
  const std::string prefix = dir + "/conda";
  const std::vector<float> data = dataset(COUNT, DIM, SEED);
  CondaRef reference(CondaSetup{COUNT, DIM, RANGE, BUILD_L, 1.2f, 750}, data.data());
  std::vector<unsigned> tags(BASE);
  for (unsigned i = 0; i < BASE; ++i) {
    tags[i] = i;
  }
  reference.insert(tags);
  // [2]
  dump(reference, data, BASE, prefix);
  pipeann::SSDIndex<float>::save_from_graph(prefix + "_base.bin", prefix + "_graph.flat",
                                            prefix + "_disk.index", RANGE);
  std::vector<float> base(data.begin(), data.begin() + (size_t) BASE * DIM);
  ExactNeighbor<float> exact(pipeann::Metric::L2, base, DIM);
  pipeann::IndexBuildParameters params;
  params.R = RANGE;
  params.L = BUILD_L;
  params.C = 750;
  params.alpha = 1.2f;
  params.beam_width = 4;
  params.num_threads = 1;
  params.max_nthreads = 4;
  params.conda = 2;
  std::shared_ptr<AlignedFileReader> reader(new LinuxAlignedFileReader());
  pipeann::SSDIndex<float> port(pipeann::Metric::L2, reader, &exact, true, &params);
  port.load(prefix.c_str(), true);
  port.condaload(prefix + "_base.bin", COUNT, reference.entry());
  port.start_bg_io();
  if (!same(reference, port, reference.locations())) {
    std::printf("load differs\n");
    return 1;
  }
  // [3]
  for (unsigned r = 0; r < ROUNDS; ++r) {
    std::vector<unsigned> gone, fresh;
    for (unsigned i = 0; i < STEP; ++i) {
      gone.push_back(r * STEP + i);
      fresh.push_back(BASE + r * STEP + i);
    }
    reference.remove(gone);
    port.condadelete(std::vector<uint32_t>(gone.begin(), gone.end()), 1);
    if (!same(reference, port, reference.locations())) {
      std::printf("round %u: delete differs\n", r);
      return 1;
    }
    reference.insert(fresh);
    port.condabatch();
    for (unsigned tag : fresh) {
      port.insert_in_place(data.data() + (size_t) tag * DIM, tag);
    }
    if (!same(reference, port, reference.locations())) {
      std::printf("round %u: insert differs\n", r);
      return 1;
    }
    std::printf("round %u: %u deletes and %u inserts identical over %u locations\n", r, STEP, STEP,
                reference.locations());
  }
  port.stop_bg_io();
  std::printf("PASS\n");
  return 0;
}
