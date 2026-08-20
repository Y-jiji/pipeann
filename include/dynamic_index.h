#pragma once

#include <omp.h>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include "index.h"
#include "linux_aligned_file_reader.h"
#include "nbr/nbr.h"
#include "utils/partition.h"
#include "utils/index_build_utils.h"
#include "ssd_index.h"
#include "utils.h"
#include <sys/sysinfo.h>

#ifdef USE_TCMALLOC
#include <gperftools/malloc_extension.h>
#endif

class BaseDynamicIndex {
 public:
  BaseDynamicIndex() = default;
  virtual ~BaseDynamicIndex() = default;
};

inline std::vector<std::shared_ptr<pipeann::AttrIndex>> wrap_native_attr_indexes(
    const std::map<uint32_t, pipeann::AttrIndex *> &base_stores) {
  std::vector<std::shared_ptr<pipeann::AttrIndex>> attr_indexes;
  attr_indexes.reserve(base_stores.size());
  for (const auto &entry : base_stores) {
    attr_indexes.push_back(std::shared_ptr<pipeann::AttrIndex>(entry.second));
  }
  return attr_indexes;
}

inline std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(const std::string &filename,
                                                                     const std::string &attr_type, uint64_t n_vectors) {
  return std::shared_ptr<pipeann::AttrIndex>(pipeann::load_attr_index_from_file(filename, attr_type, n_vectors));
}

template<class T>
class DynamicIndex : public BaseDynamicIndex {
  static constexpr float kMemIndexP = 0.01;
  static constexpr uint32_t kBuildThreshold = 100000;
  // Beam width for query search, i.e. how many records a hop reads at once.
  // OdinANN (FAST '26) evaluates at 4; that cost ~10% throughput here, but
  // only through the SQPOLL stall, so measure again at 8. Insert is
  // unaffected, its internal search takes params.beam_width instead.
  static constexpr uint32_t kSearchBeamWidth = 8;
  using TagT = uint32_t;

 public:
  using data_type = T;

 public:
  DynamicIndex() = delete;

  explicit DynamicIndex(uint32_t data_dim, pipeann::Metric metric, pipeann::IndexBuildParameters *params = nullptr)
      : data_dim_(data_dim), metric_(metric) {
    if (params != nullptr) {
      params_ = *params;
    }
    reader_.reset(new LinuxAlignedFileReader());
    nbr_handler_ = new pipeann::PQNeighbor<T>(metric_);
    mem_index_.reset(new pipeann::Index<T, TagT>(metric_, data_dim_));
    disk_index_.reset(new pipeann::SSDIndex<T, TagT>(metric_, reader_, nbr_handler_, true, params));

    mem_index_params_.set(32, 64, 750, 1.2, params_.num_threads, true);
    cur_index_prefix_ = "./test";
    // rand value for updating in-memory index.
    std::random_device rd;
    gen = std::mt19937(rd());
  }

  float rand_uniform() {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(gen);
  }

  ~DynamicIndex() {
    if (use_mem_index_for_disk_index_ && disk_index_) {
      disk_index_->mem_index_.release();
    }
  }

  // Releases the calling thread's io_uring context. An ephemeral worker
  // pool must call this at each worker's exit, or the thread_local ring's
  // fds leak (raw-pointer TLS runs no destructor on thread exit).
  void deregister_thread() { reader_->deregister_thread(); }

  // Load an index from disk.
  // If copy_to_shadow is true, the disk index is first copied to a shadow prefix
  // to avoid polluting the original index during in-place inserts.
  void load(const std::string &index_prefix, bool copy_to_shadow = false) {
    auto mem_index_path = index_prefix + "_mem.index";
    if (file_exists(mem_index_path)) {
      use_mem_index_for_disk_index_ = true;
      mem_index_.reset(pipeann::SSDIndex<T, TagT>::load_to_mem(mem_index_path, metric_));
    } else {
      use_mem_index_for_disk_index_ = false;
    }

    auto disk_index_file = index_prefix + "_disk.index";
    if (file_exists(disk_index_file)) {
      use_disk_index_ = true;

      std::string load_prefix = index_prefix;
      if (copy_to_shadow) {
        std::string shadow_prefix = index_prefix + "_shadow";
        disk_index_->copy_index(index_prefix, shadow_prefix);
        LOG(INFO) << "Copy disk index file to " << shadow_prefix << "_disk.index";
        load_prefix = shadow_prefix;
      }

      disk_index_->load(load_prefix.c_str(), true); // enable writes.
      if (use_mem_index_for_disk_index_) {
        // mem_index is disk_index's navigation graph.
        disk_index_->mem_index_.reset(mem_index_.get());
      }
      mem_L = use_mem_index_for_disk_index_ ? 10 : 0;
      data_dim_ = disk_index_->meta_.data_dim;
      cur_index_prefix_ = load_prefix;
    } else {
      use_disk_index_ = false;
      cur_index_prefix_ = index_prefix;
    }
  }

  void transform_mem_index_to_disk_index() {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);
    LOG(INFO) << "Transform memory index to disk index.";

    const std::string data_path = cur_index_prefix_ + "_mem_data.bin";
    pipeann::save_bin<T>(data_path, mem_index_->_data.data(), mem_index_->_nd, mem_index_->_dim);
    LOG(INFO) << "Memory index data saved to " << data_path;

    const std::string tags_path = cur_index_prefix_ + "_disk.index.tags";
    mem_index_->save_tags(tags_path);

    // re-build. Forward R/L from params_ so the initial disk build matches
    // the parameters the caller wanted for inserts (used by the insert-only
    // benchmark; default 0 preserves the auto-configured path). Build the
    // in-memory entry-point index too, so the following load() finds
    // _mem.index and sets mem_L, matching a prebuilt index's search path.
    this->build(data_path, cur_index_prefix_, tags_path.c_str(), true, params_.R, params_.L);
    this->load(cur_index_prefix_, true);  // here sets use_disk_index_ to true.
    LOG(INFO) << "Transform memory index to disk index done.";
  }

  // Use file path to build.
  void build(const std::string &data_path, const std::string &index_prefix, const char *tag_file = nullptr,
             bool build_mem_index = false, uint32_t max_nbrs = 0, uint32_t build_L = 0, uint32_t PQ_bytes = 0,
             uint32_t memory_use_GB = 0, pipeann::AttrWriter *attr_writer = nullptr, uint32_t range_dense = 0,
             const std::string &train_query_path = "", uint32_t R_ood = 0, uint32_t L_ood = 1500) {
    // automatically configure max_nbrs.
    size_t nr = 0, nc = 0;  // nr is number of points, nc is dimension.
    pipeann::get_bin_metadata(data_path, nr, nc);
    int nr_log10 = std::min(9, (int) std::ceil(std::log10(nr)));

    if (max_nbrs == 0) {
      uint32_t nr_nbr_arr[] = {64, 64, 64, 64, 64, 64, /* 1M */ 64, /* 10M */ 64, /* 100M */ 96, /* 1B */ 128};
      max_nbrs = nr_nbr_arr[nr_log10];
      LOG(INFO) << "Dataset contains " << nr << " points. Setting max_nbrs to " << max_nbrs;
    }

    if (PQ_bytes == 0) {
      PQ_bytes = nc / 4;                    // experience value.
      PQ_bytes = std::max(PQ_bytes, 32u);   // at least 32 bytes.
      PQ_bytes = std::min(PQ_bytes, 128u);  // at most 128 bytes.
      LOG(INFO) << "Data dimension is " << nc << ". Setting PQ_bytes to " << PQ_bytes;
    }

    if (build_L == 0) {
      build_L = max_nbrs + 32;
    }

    if (memory_use_GB == 0) {
      struct sysinfo info;
      sysinfo(&info);
      memory_use_GB = info.totalram / (1024 * 1024 * 1024) * 3 / 4;
      LOG(INFO) << "Memory use not specified. Using 75% of total memory: " << memory_use_GB << "GB";
    }

    // densify by filling one more page of dense neighbors.
    // Only for filtered vector search with >= 1M vectors.
    if (attr_writer != nullptr && nr >= 1000000 && range_dense == 0) {
      uint32_t estimated_node_size = nc * sizeof(T) + attr_writer->attr_size() + (1 + max_nbrs) * sizeof(uint32_t);
      uint32_t gap = ROUND_UP(estimated_node_size, SECTOR_LEN) - estimated_node_size;
      range_dense = gap / sizeof(uint32_t) / 100 * 100;  // make it a multiple of 100 :).
      if (range_dense <= 900) {                          // <= 3.5KB free space, use one more page.
        range_dense += 1000;
      }
    }

    if (train_query_path != "" && R_ood == 0) {
      R_ood = max_nbrs / 2;
      LOG(INFO) << "Automatically setting #neighbors for OOD to " << R_ood;
    }

    pipeann::build_disk_index<T, TagT>(data_path.c_str(), index_prefix.c_str(), max_nbrs, build_L, memory_use_GB,
                                       params_.num_threads, PQ_bytes, metric_, tag_file, nbr_handler_, attr_writer,
                                       range_dense, 0, train_query_path, R_ood, L_ood);

    if (build_mem_index) {
      build_mem(data_path, index_prefix);
    }
  }

  void set_index_prefix(const std::string &index_prefix) {
    cur_index_prefix_ = index_prefix;
  }

  void omp_set_num_threads(uint32_t num_threads) {
    ::omp_set_num_threads(std::min(num_threads, std::thread::hardware_concurrency()));
  }

  size_t npoints() const {
    if (use_disk_index_) {
      return disk_index_->cur_id.load();
    }
    return mem_index_->get_num_points();
  }

  std::pair<pipeann::Selector *, std::vector<pipeann::Attributes>> load_filter_from_json(
      const std::string &config_path) {
    if (!use_disk_index_) {
      throw std::runtime_error("Native selector requires a loaded disk index");
    }

    auto base_stores = pipeann::load_base_attr_from_config(config_path, disk_index_->meta_.npoints);

    auto attr_indexes = wrap_native_attr_indexes(base_stores);
    native_attr_indexes_.insert(native_attr_indexes_.end(), attr_indexes.begin(), attr_indexes.end());
    attr_index_map_.clear();
    for (const auto &[key, attr_index] : base_stores) {
      attr_index_map_[key] = attr_index;
    }
    auto [selector, query_attrs] = pipeann::load_selector_from_config(config_path, base_stores);
    return {selector, std::move(query_attrs)};
  }

  std::shared_ptr<pipeann::AttrIndex> load_attr_index_from_file(uint32_t key, const std::string &filename,
                                                                const std::string &attr_type) {
    if (!use_disk_index_) {
      throw std::runtime_error("Native attr index requires a loaded disk index");
    }

    auto attr_index = ::load_attr_index_from_file(filename, attr_type, disk_index_->meta_.npoints);
    native_attr_indexes_.push_back(attr_index);
    attr_index_map_[key] = attr_index.get();
    return attr_index;
  }

  void build_mem(const std::string &data_path, const std::string &index_prefix) {
    // sample rate 0.01
    std::string sample_prefix = index_prefix + "_mem_sample";
    gen_random_slice<T>(data_path, sample_prefix, kMemIndexP);

    std::string sample_data_bin = sample_prefix + "_data.bin";
    uint64_t data_num, data_dim;
    pipeann::get_bin_metadata(sample_data_bin, data_num, data_dim);

    std::string sample_id_bin = sample_prefix + "_ids.bin";
    std::ifstream reader;
    reader.open(sample_id_bin, std::ios::binary);
    reader.seekg(2 * sizeof(uint32_t), std::ios::beg);
    uint32_t tags_size = data_num * data_dim;
    std::vector<TagT> tags(data_num);
    reader.read((char *) tags.data(), tags_size * sizeof(uint32_t));
    reader.close();

    auto s = std::chrono::high_resolution_clock::now();
    mem_index_.reset(new pipeann::Index<T, TagT>(metric_, data_dim));
    // We should normalize_cosine here, as data is not pre-normalized.
    mem_index_->build(sample_data_bin.c_str(), data_num, mem_index_params_, tags);
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

    LOG(INFO) << "Finish building memory index, indexing time: " << diff.count() << "\n";
    std::string save_path = index_prefix + "_mem.index";
    pipeann::SSDIndex<T, TagT>::save_from_mem(*mem_index_, save_path);
  }

  // Single-query search. Returns the number of valid results written.
  // When `range` is finite, routes to range_search and pads unused output
  // slots with sentinels (UINT32_MAX / FLT_MAX).
  size_t search(const T *query, uint32_t topk, uint32_t L, TagT *out_ids, float *out_dists,
                pipeann::QueryStats *stats = nullptr, pipeann::Selector *selector = nullptr,
                const pipeann::Attributes *query_attrs = nullptr,
                float range = std::numeric_limits<float>::infinity()) {
    std::vector<TagT> tags(L);
    std::vector<float> distances(L);
    const bool use_range = !std::isinf(range);
    size_t n_valid = L;

    if (selector != nullptr) {
      disk_index_->spec_filter_search(query, L, L, selector, *query_attrs, tags.data(), distances.data(),
                                      kSearchBeamWidth, stats);
    } else if (use_range) {
      n_valid =
          disk_index_->range_search(query, range, tags.data(), distances.data(), kSearchBeamWidth, mem_L, L, stats);
    } else if (use_disk_index_) {
      disk_index_->pipe_search(query, L, mem_L, L, tags.data(), distances.data(), kSearchBeamWidth, stats);
    } else {
      mem_index_->search_with_tags(query, L, L, tags.data(), distances.data());
    }

    size_t pos = 0;
    for (size_t j = 0; j < n_valid && pos < topk; j++) {
      if (this->deleted_nodes_set_.find(tags[j]) == this->deleted_nodes_set_.end()) {
        out_ids[pos] = tags[j];
        out_dists[pos] = distances[j];
        pos++;
      }
    }
    if (use_range) {
      for (size_t j = pos; j < topk; j++) {
        out_ids[j] = std::numeric_limits<TagT>::max();
        out_dists[j] = std::numeric_limits<float>::max();
      }
    }
    return pos;
  }

  // Batch search (for Python). Writes results to out_ids and out_dists arrays (n_queries * topk each).
  void search(const T *queries, uint32_t n_queries, uint32_t topk, uint32_t L, TagT *out_ids, float *out_dists,
              pipeann::Selector *selector = nullptr,
              const std::vector<pipeann::Attributes> &query_attrs = std::vector<pipeann::Attributes>(),
              float range = std::numeric_limits<float>::infinity()) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);

    if ((selector != nullptr || !std::isinf(range)) && !use_disk_index_) {
      LOG(ERROR) << "Filtered/range search requires a loaded disk index";
      size_t total = static_cast<size_t>(n_queries) * static_cast<size_t>(topk);
      std::fill_n(out_ids, total, TagT());
      std::fill_n(out_dists, total, 0.0f);
      return;
    }

    if (selector != nullptr && query_attrs.size() != static_cast<size_t>(n_queries)) {
      throw std::runtime_error("Number of query attrs must match number of queries");
    }

#pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < n_queries; i++) {
      search(queries + i * data_dim_, topk, L, out_ids + i * topk, out_dists + i * topk, nullptr, selector,
             selector != nullptr ? &query_attrs[i] : nullptr, range);
    }
  }

  // Bulk add.
  void add(const T *vectors, const TagT *tags, uint32_t n_vectors,
           const std::vector<pipeann::Attributes> *attrs_vec = nullptr) {
    save_mu_.lock_shared();

#pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < n_vectors; i++) {
      const T *vector_p = vectors + i * data_dim_;
      const pipeann::Attributes *attr_p = (attrs_vec != nullptr) ? &(*attrs_vec)[i] : nullptr;
      do_insert(vector_p, tags[i], attr_p);
    }
    save_mu_.unlock_shared();

    if (!use_disk_index_ && mem_index_->get_num_points() > kBuildThreshold) {
      transform_mem_index_to_disk_index();
    }
  }

  // Single-point insert (for C++ benchmarks / streaming updates). `stats`,
  // if given, is populated with the disk-path search stats (n_ios, n_cmps,
  // total_us, ...) of the insert's internal do_pipe_search call -- null
  // (mem-index-only) inserts leave it untouched.
  int insert(const T *point, const TagT &tag, const pipeann::Attributes *attrs = nullptr,
             pipeann::QueryStats *stats = nullptr, std::uint64_t *pages = nullptr,
             std::uint64_t *close = nullptr, std::uint64_t begun = 0) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    do_insert(point, tag, attrs, stats, pages, close, begun);
    return 0;
  }

  void start_bg_io() {
    if (use_disk_index_) disk_index_->start_bg_io();
  }

  void stop_bg_io() {
    if (use_disk_index_) disk_index_->stop_bg_io();
  }

  // Single-point lazy delete (marks tag for future merge).
  void lazy_delete(const TagT &tag) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
      deleted_nodes_set_.insert(tag);
      deleted_nodes_.push_back(tag);
    }
    mem_index_->lazy_delete(tag);
  }

  // Bulk remove.
  void remove(const TagT *tags, uint32_t n_tags) {
    auto mu = std::shared_lock<std::shared_mutex>(save_mu_);
    for (uint32_t i = 0; i < n_tags; i++) {
      TagT tag = tags[i];
      if (deleted_nodes_set_.find(tag) == deleted_nodes_set_.end()) {
        deleted_nodes_set_.insert(tag);
        deleted_nodes_.push_back(tag);
      }
      mem_index_->lazy_delete(tag);
    }
  }

  // Merge in-place inserts and deletions into the disk index, then save to index_prefix.
  // If index_prefix == cur_index_prefix_, uses a double-version strategy to avoid corrupting the live index.
  bool save(const std::string &index_prefix, uint32_t nthreads = 0) {
    auto mu = std::lock_guard<std::shared_mutex>(save_mu_);

    if (cur_index_prefix_.empty()) {
      LOG(ERROR) << "Current index prefix is empty. Cannot save.";
      return false;
    }

    uint32_t merge_threads = (nthreads > 0) ? nthreads : omp_get_max_threads();

    if (use_disk_index_) {
      std::string out_prefix = index_prefix;
      if (cur_index_prefix_ == index_prefix) {
        out_prefix = index_prefix + "_v2";
        LOG(INFO) << "Saving to the same index prefix. Using double-version: " << out_prefix;
      }

      auto id_map =
          disk_index_->merge_deletes(cur_index_prefix_, out_prefix, deleted_nodes_, deleted_nodes_set_, merge_threads);

      // Merge attr_indexes.
      for (auto &[key, attr_index] : attr_index_map_) {
        attr_index->merge(id_map);
      }

      disk_index_->reload(out_prefix.c_str());

      if (cur_index_prefix_ == index_prefix) {
        LOG(INFO) << "Copying index from " << out_prefix << " to " << index_prefix;
        disk_index_->copy_index(out_prefix, index_prefix);
        // First reload released merge_lock; re-acquire to pair with the unlock in this reload.
        disk_index_->merge_lock.lock();
        disk_index_->reload(index_prefix.c_str());
      }

      cur_index_prefix_ = index_prefix;
      deleted_nodes_.clear();
      deleted_nodes_set_.clear();
    }

    if (!use_disk_index_ || use_mem_index_for_disk_index_) {
      pipeann::SSDIndex<T, TagT>::save_from_mem(*mem_index_, index_prefix + "_mem.index");
    }
#ifdef USE_TCMALLOC
    MallocExtension::instance()->ReleaseFreeMemory();
#endif
    return true;
  }

  std::string to_string() const {
    return std::string("DynamicIndex<") + typeid(T).name() + ">";
  }

  const std::string &index_prefix() const {
    return cur_index_prefix_;
  }

 private:
  void do_insert(const T *point_p, TagT tag, const pipeann::Attributes *attrs = nullptr,
                 pipeann::QueryStats *stats = nullptr, std::uint64_t *pages = nullptr,
                 std::uint64_t *close = nullptr, std::uint64_t begun = 0) {
    if (use_disk_index_) {
      int target_id = disk_index_->insert_in_place(point_p, tag, attrs, stats, pages, close, begun);

      // Insert attributes into attr_indexes (if loaded).
      if (attrs != nullptr) {
        for (const auto &[key, attr] : attrs->attrs_) {
          auto it = attr_index_map_.find(key);
          if (it == attr_index_map_.end()) {
            LOG(ERROR) << "Attribute key " << key << " not found in attr_index_map_. Skipping attribute insertion.";
            continue;
          }
          it->second->insert(target_id, attr);
        }
      }

      if (use_mem_index_for_disk_index_ && rand_uniform() <= kMemIndexP) {  // probably insert into the memory index.
        mem_index_->insert_point(point_p, mem_index_params_, target_id);
      }
    } else {
      mem_index_->insert_point(point_p, mem_index_params_, tag);
    }
  }

  bool use_disk_index_ = false;
  bool use_mem_index_for_disk_index_ = false;
  std::shared_ptr<AlignedFileReader> reader_;
  pipeann::AbstractNeighbor<T> *nbr_handler_;
  std::shared_mutex save_mu_;  // save mutex.
  std::string data_path_;
  std::string cur_index_prefix_;
  uint32_t data_dim_;
  pipeann::Metric metric_;
  std::vector<TagT> deleted_nodes_;
  tsl::robin_set<TagT> deleted_nodes_set_;  // copy of deleted nodes.

  // if vectors are less than the threshold, use mem index instead.
  std::mt19937 gen;
  pipeann::IndexBuildParameters mem_index_params_;
  pipeann::IndexBuildParameters params_;
  std::shared_ptr<pipeann::Index<T, TagT>> mem_index_;
  std::shared_ptr<pipeann::SSDIndex<T, TagT>> disk_index_;
  std::vector<std::shared_ptr<pipeann::AttrIndex>> native_attr_indexes_;
  std::map<uint32_t, pipeann::AttrIndex *> attr_index_map_;  // non-owning key -> attr_index mapping
  uint32_t mem_L = 0;                                        // memory search L in pipe search.
};
