#pragma once
#include <immintrin.h>
#include <cassert>
#include <cstdint>
#include <limits>
#include <string>
#include <set>
#include <omp.h>

#include <memory>

#include "aligned_file_reader.h"
#include "ssd_index_defs.h"
#include "filter/attribute.h"
#include "filter/selector.h"
#include "utils/concurrent_queue.h"
#include "utils/lock_table.h"
#include "utils/percentile_stats.h"
#include "utils/rwlock.h"
#include "utils/tsl/robin_map.h"
#include "nbr/abstract_nbr.h"
#include "utils.h"
#include "index.h"

enum SearchMode { BEAM_SEARCH = 0, PAGE_SEARCH = 1, PIPE_SEARCH = 2, CORO_SEARCH = 3 };

namespace pipeann {
  template<typename T, typename TagT = uint32_t>
  class SSDIndex {
   public:
    static constexpr uint32_t kAllocatedID = std::numeric_limits<uint32_t>::max() - 1;
    static constexpr uint32_t kExpandedNodesFactor = 10;

    std::unique_ptr<Index<T, uint32_t>> mem_index_;  // in-memory navigation graph

    // Index metadata (consolidated).
    SSDIndexMetadata<T> meta_;

    // For updates:
    // meta.npoints is the index's initial size (constant between two merges).
    // cur_id is the ID to be allocated (+1 for each insert, starting from meta.npoints)
    // cur_loc is the tail of the index file, which will be greater than cur_id if overprovisioned.
    // - This is because holes may exist in the index for update combining.
    std::atomic<uint64_t> cur_id, cur_loc;

    SSDIndex(pipeann::Metric m, std::shared_ptr<AlignedFileReader> &file_reader, AbstractNeighbor<T> *nbr = nullptr,
             bool tags = false, IndexBuildParameters *parameters = nullptr);

    ~SSDIndex();

    // Use node_from_page() to create instances for DiskNode.
    // Params: location (id2loc first if using ID), the page-aligned buffer.
    // The offset is calculated in the constructor.
    inline DiskNode<T> node_from_page(char *page_buf, uint32_t loc) {
      return DiskNode<T>(page_buf, loc, meta_);
    }

    // Unaligned offset to location.
    inline uint64_t u_loc_offset(uint64_t loc) const {
      return meta_.u_loc_offset(loc);
    }

    inline uint64_t u_loc_offset_nbr(uint64_t loc) const {
      return meta_.u_loc_offset_nbr(loc);
    }

    inline uint64_t loc_sector_no(uint64_t loc) const {
      return meta_.loc_sector_no(loc);
    }

    inline uint64_t sector_to_loc(uint64_t sector_no, uint32_t sector_off) const {
      return meta_.sector_to_loc(sector_no, sector_off);
    }

    // Trace-driver access: score every vertex in [0, n) against `query` in
    // PQ space, into `out`. A search scores only the neighbours it happens to
    // meet, so a trace captured here could only ever be replayed along this
    // run's own path; the whole vector lets a replay follow the graph
    // anywhere and diverge visibly. No behaviour of its own.
    void estimate_all(const T *query, uint64_t n, float *out) {
      QueryBuffer *buf = pop_query_buf(query);
      buf->reset();
      nbr_handler->initialize_query(buf->template aligned_query<T>(), buf);
      std::vector<uint32_t> ids(MAX_N_EDGES);
      for (uint64_t i = 0; i < n; i += MAX_N_EDGES) {
        const uint64_t m = std::min<uint64_t>(MAX_N_EDGES, n - i);
        for (uint64_t j = 0; j < m; j++) {
          ids[j] = static_cast<uint32_t>(i + j);
        }
        nbr_handler->compute_dists(buf, ids.data(), m);
        memcpy(out + i, buf->aligned_dist_scratch, m * sizeof(float));
      }
      push_query_buf(buf);
    }

    void init_metadata(const SSDIndexMetadata<T> &meta) {
      meta.print();
      this->meta_ = meta;
      this->cur_id = this->cur_loc = meta.npoints;
      this->aligned_dim = ROUND_UP(meta_.data_dim, 8);
      this->params.R = meta.range;
      this->io_size = meta.io_size();
      this->io_size_dense = meta.io_size_dense();
      LOG(INFO) << "IO size (normal): " << io_size << " IO size (dense): " << io_size_dense;

      // Aligned.
      if (meta_.nnodes_per_sector != 0 && meta_.npoints % meta_.nnodes_per_sector != 0) {
        cur_loc += meta_.nnodes_per_sector - (meta_.npoints % meta_.nnodes_per_sector);
      }
      LOG(INFO) << "Cur location: " << this->cur_loc;

      // Update-related metadata, if not initialized in constructor, initialize here.
      if (params.L == 0) {
        // Experience values.
        LOG(INFO) << "Automatically set the update-related parameters.";
        params.set(meta.range, meta.range + 32, 384, 1.2, 0, true, 8);
        params.print();
      }
    }

    void init_query_buf(QueryBuffer &buf) {
      pipeann::alloc_aligned((void **) &buf.coord_scratch_, this->aligned_dim * sizeof(T), 8 * sizeof(T));
      buf.sector_scratch = (char *) this->reader->alloc_io_buf(MAX_N_SECTOR_READS * io_size_dense, SECTOR_LEN);
      pipeann::alloc_aligned((void **) &buf.nbr_vec_scratch,
                             MAX_N_EDGES * AbstractNeighbor<T>::MAX_BYTES_PER_NBR * sizeof(uint8_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_id_scratch, MAX_N_EDGES * sizeof(uint32_t), 256);
      pipeann::alloc_aligned((void **) &buf.nbr_ctx_scratch, ROUND_UP(nbr_handler->query_ctx_size(), 256), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_dist_scratch, MAX_N_EDGES * sizeof(float), 256);
      pipeann::alloc_aligned((void **) &buf.aligned_query_, this->aligned_dim * sizeof(T), 8 * sizeof(T));

      buf.visited = new tsl::robin_set<uint64_t>(4096);
      buf.page_visited = new tsl::robin_set<unsigned>(4096);

      memset(buf.sector_scratch, 0, MAX_N_SECTOR_READS * io_size_dense);
      memset(buf.coord_scratch_, 0, this->aligned_dim * sizeof(T));
      memset(buf.aligned_query_, 0, this->aligned_dim * sizeof(T));
    }

    QueryBuffer *pop_query_buf(const T *query) {
      QueryBuffer *data = this->thread_data_queue.pop();
      while (data == this->thread_data_queue.null_T) {
        this->thread_data_queue.wait_for_push_notify();
        data = this->thread_data_queue.pop();
      }

      if (likely(query != nullptr)) {
        if (this->metric == Metric::COSINE) {
          // Data has been normalized. Normalize search vector too.
          pipeann::normalize_data(data->aligned_query<T>(), query, meta_.data_dim);
        } else {
          memcpy(data->aligned_query<T>(), query, meta_.data_dim * sizeof(T));
        }
      }
      return data;
    }

    void push_query_buf(QueryBuffer *data) {
      this->thread_data_queue.push(data);
      this->thread_data_queue.push_notify_all();
    }

    // Load compressed data, and obtains the handle to the disk-resident index.
    // When write is enabled, SPDK is forced to copy the disk index, regardless of marker.
    // For other io engines, enable_writes do nothing.
    int load(const char *index_prefix, bool enable_writes);

    // Load the companion nav graph at {mem_index_path} (SSD-format disk file) into mem_index_.
    void load_mem_index(const std::string &mem_index_path);

    // Write an in-memory Index to an SSD-format disk file at {disk_file}, plus {disk_file}.tags.
    // If target_dense_width > mem.range, sample (target_dense_width - mem.range) 2-hop
    // neighbors per node on the fly and store them as dense edges; otherwise only
    // 1-hop edges are written.
    static void save_from_mem(Index<T, TagT> &mem, const std::string &disk_file, uint16_t target_dense_width = 0,
                              AttrWriter *attr_writer = nullptr);

    // Empty neighbor slot in a graph file handed to save_from_graph.
    static constexpr uint32_t kGraphPad = 0xFFFFFFFF;

    // Write an SSD-format disk file at {disk_file} from a base file and a graph built elsewhere,
    // streaming both so neither is ever held whole. {base_file} is a bin file (uint32 npts,
    // uint32 dim, then packed coords) and {graph_file} a headerless npts * range matrix of uint32
    // neighbor ids, rows in id order, unused slots holding kGraphPad. The entry point is the node
    // nearest the centroid, as Index::calculate_entry_point picks it. No tags file is written:
    // ids and tags coincide, which is what id2tag falls back to.
    static void save_from_graph(const std::string &base_file, const std::string &graph_file,
                                const std::string &disk_file, uint32_t range);

    // Read an SSD-format disk file at {disk_file} (+ {disk_file}.tags if present) into an Index.
    // Caller owns the returned pointer.
    static Index<T, TagT> *load_to_mem(const std::string &disk_file, Metric metric);

    // Read-only search algorithms (static datasets only).
    size_t coro_search(T **queries, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT **res_tags, float **res_dists, const uint64_t beam_width, int N);

    size_t page_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    // Search supporting updates (with concurency control).
    size_t beam_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t pipe_search(const T *query, const uint64_t k_search, const uint32_t mem_L, const uint64_t l_search,
                       TagT *res_tags, float *res_dists, const uint64_t beam_width, QueryStats *stats = nullptr);

    // Range search: return every tag whose distance to query is <= range (in user-facing metric).
    // Early stop when all vectors with PQ_distance > kRangeEarlyStopFactor * range are explored, may miss some results.
    // Caller pre-allocates res_tags/res_dists to at least l_search entries.
    size_t range_search(const T *query, const float range, TagT *res_tags, float *res_dists, const uint64_t beam_width,
                        const uint32_t mem_L = 0, const uint64_t l_search = 8192, QueryStats *stats = nullptr);

    size_t spec_filter_search(const T *query, const uint64_t k_search, const uint64_t l_search, Selector *selector,
                              const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                              const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t spec_prefilter_search(const T *query, const uint64_t k_search, const uint64_t l_search, Selector *selector,
                                 const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                 const uint64_t beam_width, QueryStats *stats = nullptr);

    size_t spec_postfilter_search(const T *query, const uint64_t k_search, const uint64_t l_search,
                                  const uint64_t l_max, Selector *selector, const Attributes &query_attrs,
                                  TagT *res_tags, float *res_dists, const uint64_t beam_width,
                                  QueryStats *stats = nullptr);

    size_t spec_infilter_search(const T *query, const uint64_t k_search, const uint64_t l_search, const uint64_t l_max,
                                Selector *selector, const Attributes &query_attrs, TagT *res_tags, float *res_dists,
                                const uint64_t beam_width, QueryStats *stats = nullptr);

    int insert_in_place(const T *point, const TagT &tag, const Attributes *attrs = nullptr, QueryStats *stats = nullptr,
                        uint64_t *pages = nullptr, uint64_t *close = nullptr, uint64_t begun = 0);

    void start_bg_io();
    void stop_bg_io();

    /**
     * @brief Write the disk index's graph as a ParlayANN graph, node and neighbors as tags
     * SHAME(TALLFUNC)
     * @param path output file: u32 nodes, u32 range, every degree, then every neighbor list
     */
    void dumpgraph(const std::string &path);

    /**
     * @brief Set the beam widths of the update searches; call while no insert or delete runs
     * @param insert sector reads an insert's search issues together, OdinANN's and CONDA's
     * @param remove sector reads a CONDA delete repair's search issues together
     */
    void set_beam_widths(uint32_t insert, uint32_t remove);

    /**
     * @brief Start CONDA over the loaded graph: every loaded id live and stored, tag i at its id
     * @param base bin file of the loaded points, read once for CONDA's dead test on point[0]
     * @param capacity bound on every id and tag the run will hold
     * @param entry CONDA's entry point
     */
    void condaload(const std::string &base, uint64_t capacity, uint32_t entry);

    /**
     * @brief Open an insert batch: its first insert runs alone and non-lazy when no deleted vertex
     *  awaits consolidation, as CONDA's BatchInsert_crng
     */
    void condabatch();

    /**
     * @brief CONDA's delete step: MarkDelete of every tag in order, the repair of each deleted vertex,
     *  then their consolidation into tombstones on the free stack
     * @param tags the deleted tags, in MarkDelete order
     * @param nthreads repair and consolidation threads
     */
    void condadelete(const std::vector<TagT> &tags, uint32_t nthreads);

    /**
     * @brief CONDA's Delete (the paper's Algorithm 4) of one tag at once, beside concurrent inserts,
     *  searches and removes: mark it, relink its in- and out-neighbors, then turn it into a tombstone
     *  on the free stack
     * @param tag the deleted tag
     * @return false if the tag is not live
     */
    bool condaremove(TagT tag);

    /**
     * @brief The current neighbor list of an id, empty when never stored
     * @param id the vertex
     * @return its out-neighbors in stored order
     */
    std::vector<uint32_t> condaedges(uint32_t id);

    // Merge deletes (NOTE: index read-only during merge.)
    // Returns id_map: old_id -> new_id.
    libcuckoo::cuckoohash_map<uint32_t, uint32_t> merge_deletes(const std::string &in_path_prefix,
                                                                const std::string &out_path_prefix,
                                                                const std::vector<TagT> &deleted_nodes,
                                                                const tsl::robin_set<TagT> &deleted_nodes_set,
                                                                uint32_t nthreads);

    // After merge, reload the index.
    void reload(const char *index_prefix);

    void write_metadata_and_pq(const std::string &in_path_prefix, const std::string &out_path_prefix,
                               std::vector<TagT> *new_tags = nullptr);

    void copy_index(const std::string &prefix_in, const std::string &prefix_out);

    /** @brief Held exclusive during the final metadata swap in merge_deletes and reload, acquired in
     *  merge_deletes and released in reload; callers that reload again (the double-version path
     *  after copy_index) re-acquire it; writer-first, so a merge's swap gets in while searches keep
     *  taking it shared, and CONDA's entry switch takes it too */
    WriterFirstMutex merge_lock;

   private:
    // Background insert I/O commit.
    struct BgTask {
      QueryBuffer *thread_data;
      std::vector<IORequest> writes;
      std::vector<uint64_t> pages_to_unlock;
      std::vector<uint64_t> pages_to_deref;
      // Counters the caller wants this task to fill: the pages it writes and
      // the raw steady_clock nanosecond at which it finished, which closes
      // the insert that queued it. Null when the caller is not logging that
      // insert. They point into storage the caller keeps alive until it
      // stop_bg_io joins, never into the task itself.
      uint64_t *pages = nullptr;
      uint64_t *close = nullptr;
      uint64_t begun = 0;
      bool terminate = false;
      // A task with no thread_data (a CONDA flush) owns the io buffer its writes point into and a condaslots
      // token, freed and returned once written.
      char *buf = nullptr;
      uint64_t buflen = 0;
    };

    /** @brief Seeding of a CONDA walk: non-lazy insert, lazy insert, or repair of a deleted vertex */
    enum class CondaSeek { FULL, LAZY, HEAL };

    /** @brief One CONDA store primitive on a list: token 'S' set, 'A' add, 'D' delete every copy,
     *  'C' clear, or 'V' the vector written into the op */
    struct CondaEdit {
      uint8_t token;
      uint32_t entry;
      std::vector<uint32_t> whole;
    };

    /** @brief One CONDA operation's view: lists and vectors it read or wrote, each record's edits
     *  in order, the records it dirtied, pages it read near, and page refs to drop at its end */
    struct CondaOp {
      tsl::robin_map<uint32_t, std::vector<uint32_t>> lists;
      tsl::robin_map<uint32_t, std::vector<T>> point;
      tsl::robin_map<uint32_t, std::vector<CondaEdit>> edits;
      std::vector<uint32_t> dirty;
      std::vector<uint64_t> hints;
      std::vector<uint64_t> pages;
    };

    using PageArr = std::vector<uint32_t>;

#ifndef PIPEANN_BG_IO_THREADS
#define PIPEANN_BG_IO_THREADS 1
#endif
    static constexpr int kBgIOThreads = PIPEANN_BG_IO_THREADS;

    // Derived/runtime metadata not stored in SSDIndexMetadata.
    uint64_t aligned_dim = 0;
    uint64_t io_size = 0;
    uint64_t io_size_dense = 0;

    // File reader and index file path.
    std::shared_ptr<AlignedFileReader> &reader;
    std::string disk_index_file;

    // Neighbor handler.
    AbstractNeighbor<T> *nbr_handler;

    // Distance comparator.
    pipeann::Metric metric;
    std::shared_ptr<Distance<T>> dist_cmp;

    // Update-related parameters.
    IndexBuildParameters params;

    // Thread-specific scratch buffers.
    ConcurrentQueue<QueryBuffer *> thread_data_queue;
    std::vector<QueryBuffer *> thread_data_bufs;  // pre-allocated thread data

    // Background I/O threads for insert.
    ConcurrentQueue<BgTask *> bg_tasks = ConcurrentQueue<BgTask *>(nullptr);
    std::thread *bg_io_thread_[kBgIOThreads]{nullptr};

    // Locking tables for concurrency control.
    pipeann::SparseLockTable<uint64_t> page_lock_table;
    pipeann::SparseLockTable<uint64_t> vec_lock_table;
    pipeann::SparseLockTable<uint64_t> idx_lock_table;

    // ID to location mapping.
    // Concurrency control is done in lock_idx.
    // Only resize should be protected.
    std::vector<uint32_t> id2loc_;
    pipeann::ReaderOptSharedMutex id2loc_resize_mu_;

    // Location to ID mapping.
    // If nnodes_per_sector >= 1, page_layout[i * nnodes_per_sector + j] is the id of the j-th node in the i-th page.
    // ElseIf nnodes_per_sector == 0, page_layout[i] is the id of the i-th node (starting from loc_sector_no(i)).
    std::vector<uint32_t> loc2id_;
    pipeann::ReaderOptSharedMutex loc2id_resize_mu_;
    std::mutex alloc_lock;
    ConcurrentQueue<uint32_t> empty_pages = ConcurrentQueue<uint32_t>(kInvalidID);

    // Tag support.
    // If ID == tag, then it is not stored.
    libcuckoo::cuckoohash_map<uint32_t, TagT> tags;

    // Flags.
    bool load_flag = false;    // already loaded.
    bool enable_tags = false;  // support for tags and dynamic indexing

    /** @brief CONDA's deleted set: ids marked deleted and not yet consolidated */
    tsl::robin_set<uint32_t> condagone;
    /** @brief CONDA's free stack of tombstoned ids, reused last in first out */
    std::vector<uint32_t> condafree;
    /** @brief CONDA's tag to id map */
    libcuckoo::cuckoohash_map<TagT, uint32_t> condaloc;
    /** @brief Per id: 0 live, 1 live with point[0] at the type's max, 2 tombstone */
    std::vector<uint8_t> condahead;
    /** @brief Per id: holds a live tag, CONDA's location_to_tag_ */
    std::vector<uint8_t> condalive;
    /** @brief Per id: has a record on disk */
    std::vector<uint8_t> stored;
    /** @brief CONDA's entry point */
    uint32_t condaep = 0;
    /** @brief CONDA's tag_lock_: id reservation and the tag maps */
    std::mutex condamu;
    /** @brief Held alone by a batch's first insert, shared by the rest */
    std::shared_mutex condagate;
    /** @brief 1 while a batch's first insert is due, 2 while it runs, else 0 */
    std::atomic<int> condafresh{0};
    /** @brief CONDA's queue slots, a token per query buffer: a flush takes one before it locks anything
     *  and the background writer returns it once that flush is written, so queued flushes stay as bounded
     *  as OdinANN's inserts, each holding its query buffer until written */
    ConcurrentQueue<uint8_t> condaslots = ConcurrentQueue<uint8_t>(0);

    void init_buffers(uint64_t nthreads);
    void destroy_buffers();

    // Load id2loc and loc2id (i.e., page_layout), to support index reordering.
    void load_page_layout(const std::string &index_prefix, const uint64_t nnodes_per_sector = 0,
                          const uint64_t num_points = 0);

    void load_tags(const std::string &tag_file, size_t offset = 0);

    // Direct insert related.
    struct InsertContext {
      tsl::robin_map<uint32_t, T *> coord_map;
      T *coord_buf;
      std::vector<uint64_t> hint_pages;
      std::vector<uint64_t> page_ref;

      InsertContext(uint64_t max_nodes, uint64_t aligned_dim) {
        coord_map.reserve(max_nodes);
        pipeann::alloc_aligned((void **) &coord_buf, max_nodes * aligned_dim * sizeof(T), 256);
      }

      ~InsertContext() {
        pipeann::aligned_free(coord_buf);
      }

      InsertContext(const InsertContext &) = delete;
      InsertContext &operator=(const InsertContext &) = delete;
      InsertContext(InsertContext &&) = delete;
      InsertContext &operator=(InsertContext &&) = delete;
    };

    void do_beam_search(const T *query, uint32_t mem_L, uint32_t l_search, const uint64_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats = nullptr,
                        InsertContext *insert_ctx = nullptr);

    /**
     * @brief The non-pipelined beam search: rounds of beam_width blocking reads, each round expanded
     *  before the next is read, so beam width 1 walks strictly best first; SHAME(TALLFUNC) SHAME(MANYARG)
     * @tparam SpecFn (id) -> bool, whether a discovered neighbor may enter the candidate list
     * @tparam VerifyFn (id, node) -> bool, called on each expansion, whether it joins full_retset
     * @param query the query vector
     * @param mem_L candidates seeded from the in-memory index, 0 for none
     * @param l_search the candidate list
     * @param beam_width reads per round
     * @param is_member_approx the neighbor filter
     * @param is_member the expansion callback
     * @param full_retset every accepted expansion at its exact distance, sorted
     * @param stats walk counters, may be null
     * @param insert_ctx coordinates and pages kept for an update, may be null
     * @param seeds candidates seeded in place of the entry point, may be null
     */
    template<typename SpecFn, typename VerifyFn>
    void beam_search_common(const T *query, uint32_t mem_L, uint64_t l_search, uint64_t beam_width,
                            SpecFn is_member_approx, VerifyFn is_member, std::vector<Neighbor> &full_retset,
                            QueryStats *stats = nullptr, InsertContext *insert_ctx = nullptr,
                            const std::vector<uint32_t> *seeds = nullptr);

    void do_pipe_search(const T *query, uint32_t mem_L, uint32_t l_search, const uint64_t beam_width,
                        std::vector<Neighbor> &expanded_nodes_info, QueryStats *stats = nullptr,
                        InsertContext *insert_ctx = nullptr);

    template<typename SpecFn, typename VerifyFn>
    void pipe_search_common(const T *query, uint32_t mem_L, uint64_t l_search, uint64_t l_pool, uint64_t beam_width,
                            uint64_t read_io_size, bool use_dense_nbrs, SpecFn is_member_approx, VerifyFn is_member,
                            std::vector<Neighbor> &full_retset, QueryStats *stats = nullptr,
                            InsertContext *insert_ctx = nullptr,
                            float range_partial = std::numeric_limits<float>::infinity(),
                            const std::vector<uint32_t> *seeds = nullptr);

    /**
     * @brief CONDA's dead test, point[0] at the type's max: tombstones and live points that start so
     * @param id the vertex
     * @return whether CONDA's lazy walk treats it as deleted
     */
    bool condadead(uint32_t id);

    /**
     * @brief Whether a vertex is out of CONDA's live set: deleted, consolidated, free or never taken
     * @param id the vertex
     * @return whether a repair skips it
     */
    bool condaout(uint32_t id);

    /**
     * @brief CONDA's reserve_location, the caller holding condamu
     * @return the top of the free stack, else the next new id
     */
    uint32_t condareserve();

    /**
     * @brief CONDA's Insert_crng under BatchInsert_crng's first-insert rule
     * @param point the vector
     * @param tag its tag
     * @param stats walk counters, may be null
     * @param pages pages written, may be null
     * @param close nanoseconds from begun to the write's end, may be null
     * @param begun steady clock at the insert's start
     * @return the id it took
     */
    int condainsert(const T *point, const TagT &tag, QueryStats *stats, uint64_t *pages, uint64_t *close,
                    uint64_t begun);

    /**
     * @brief CONDA's Process_crng: relink a deleted vertex's in-neighbors and out-neighbors; SHAME(TALLFUNC)
     * @param id the deleted vertex
     */
    void condarepair(uint32_t id);

    /**
     * @brief CONDA's consolidation of one deleted vertex: empty list, max vector
     * @param id the deleted vertex
     */
    void condaclear(uint32_t id);

    /**
     * @brief CONDA's greedy walk, seeded and with its lazy rule, run by the pipelined search;
     *  SHAME(TALLFUNC)
     * @param self the inserted or deleted vertex, never walked through
     * @param seek seeding and laziness
     * @param listn candidate list
     * @param op the operation, its vector at self
     * @param expanded every expanded vertex in expansion order, at CONDA's distance to self
     * @param stats walk counters, may be null
     */
    void condasearch(uint32_t self, CondaSeek seek, uint32_t listn, CondaOp &op, std::vector<Neighbor> &expanded,
                     QueryStats *stats);

    /**
     * @brief CONDA's get_distance between two vertices
     * @param first one vertex
     * @param second the other
     * @param op the operation, exact from its vectors when it read both outside the idealized mode
     * @return squared L2
     */
    float condadist(uint32_t first, uint32_t second, CondaOp &op);

    /**
     * @brief CONDA's GetNeighbors in the operation's view, read once from disk
     * @param id the vertex
     * @param op the operation
     * @return its list, valid until the op next gains a list
     */
    const std::vector<uint32_t> &condaget(uint32_t id, CondaOp &op);

    /**
     * @brief Read the records of ids the op has not seen, a chunk of reads at a time
     * @param ids the vertices
     * @param op the operation
     */
    void condafetch(const std::vector<uint32_t> &ids, CondaOp &op);

    /**
     * @brief Apply one store primitive to the op's view and log it for the write
     * @param id the vertex
     * @param edit the primitive
     * @param op the operation
     */
    void condaedit(uint32_t id, CondaEdit edit, CondaOp &op);

    /**
     * @brief Insert_crng's CRNG choice over the sorted walk, rewiring each occluder to self
     * @param self the inserted vertex
     * @param expanded the walk sorted by distance
     * @param op the operation
     * @param pruned per candidate its occluder, or UINT_MAX
     * @param result the chosen neighbors
     * @return the index of the last chosen candidate
     */
    uint32_t condaselect(uint32_t self, const std::vector<Neighbor> &expanded, CondaOp &op,
                         std::vector<uint32_t> &pruned, std::vector<uint32_t> &result);

    /**
     * @brief InterInsert_with_expand: back edges to self, occluded candidates retried when one is refused
     * @param self the inserted vertex
     * @param expanded the walk sorted by distance
     * @param pruned per candidate its occluder, or UINT_MAX
     * @param maxid the index of the last chosen candidate
     * @param op the operation
     */
    void condalink(uint32_t self, const std::vector<Neighbor> &expanded, std::vector<uint32_t> &pruned,
                   uint32_t maxid, CondaOp &op);

    /**
     * @brief CONDA's alpha-RNG prune that reports whether self survived
     * @param center the list's owner
     * @param pool candidates at their distance to center
     * @param self the vertex whose survival is reported
     * @param result the kept ids
     * @param op the operation
     * @return whether self was kept
     */
    bool condaalpha(uint32_t center, std::vector<Neighbor> &pool, uint32_t self, std::vector<uint32_t> &result,
                    CondaOp &op);

    /**
     * @brief CONDA's sync_prune_crng, deleting each non-monotonic two-hop edge it keeps around
     * @param center the list's owner
     * @param pool candidates at their distance to center
     * @param result the kept ids
     * @param op the operation
     */
    void condacrng(uint32_t center, std::vector<Neighbor> &pool, std::vector<uint32_t> &result, CondaOp &op);

    /**
     * @brief The CONDA_LINKS pool members closest to self, self excluded
     * @param pool candidate ids
     * @param self the vertex
     * @param op the operation
     * @return up to CONDA_LINKS ids, closest first
     */
    std::vector<uint32_t> condaclose(const std::vector<uint32_t> &pool, uint32_t self, CondaOp &op);

    /**
     * @brief Write the op's dirty records relocated onto shared pages, each re-read under its lock
     *  and its edits replayed on it, the writes left to the background writer; SHAME(TALLFUNC)
     * @param op the operation
     * @param pages set by the background writer to the pages it wrote, null for none
     * @param close set by the background writer to the steady_clock ns past begun when written
     * @param begun steady_clock ns at which the caller's insert began
     * @return pages written
     */
    uint64_t condaflush(CondaOp &op, uint64_t *pages = nullptr, uint64_t *close = nullptr,
                        uint64_t begun = 0);

    // Background I/O thread function.
    void bg_io_thread();

    int get_vector_by_id(const uint32_t &id, T *vector);

    // ID, loc, page mapping.
    TagT id2tag(uint32_t id);

    // Deduplicate sorted results and copy top-k to output arrays.
    size_t copy_top_k(const std::vector<Neighbor> &sorted_results, uint64_t k, TagT *res_tags, float *res_dists,
                      float max_dist = std::numeric_limits<float>::infinity());

    uint32_t id2loc(uint32_t id);
    void set_id2loc(uint32_t id, uint32_t loc);
    uint64_t id2page(uint32_t id);

    uint32_t loc2id(uint32_t loc);
    void set_loc2id(uint32_t loc, uint32_t id);
    void erase_loc2id(uint32_t loc);

    PageArr get_page_layout(uint32_t page_no);

    void erase_and_set_loc(const std::vector<uint64_t> &old_locs, const std::vector<uint64_t> &new_locs,
                           const std::vector<uint32_t> &new_ids);
    // Returns <loc, need_read>.
    std::vector<uint64_t> alloc_loc(int n, const std::vector<uint64_t> &hint_pages,
                                    std::set<uint64_t> &page_need_to_read);
    void verify_id2loc();

    // lock-related.
    void lock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                  const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_vec(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                    const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> get_to_lock_idx(uint32_t target, const std::vector<uint32_t> &neighbors);

    std::vector<uint32_t> lock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, uint32_t target,
                                   const std::vector<uint32_t> &neighbors, bool rd = false);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const std::vector<uint32_t> &to_lock);
    void unlock_idx(pipeann::SparseLockTable<uint64_t> &lock_table, const uint32_t &to_lock);
  };
}  // namespace pipeann
