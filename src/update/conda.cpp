#include "aligned_file_reader.h"
#include "ssd_index.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>

#include <omp.h>
#include "utils/timer.h"
#include "utils/tsl/robin_map.h"
#include "utils/tsl/robin_set.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"
#include "../search/pipe_search_common.h"
#include "../search/beamwalk.h"

namespace pipeann {
  /** @brief Candidate list of CONDA's repair walk around a deleted vertex */
  constexpr uint32_t CONDA_LIST = 128;
  /** @brief Nearest walk candidates a repair links from, deleted ones then dropped */
  constexpr uint32_t CONDA_POOL = 50;
  /** @brief Pool members each in-neighbor, or each out-neighbor's new holder, is linked with */
  constexpr uint32_t CONDA_LINKS = 3;
  /** @brief Requests one reader call carries at most, the io_uring ring depth */
  constexpr uint32_t CONDA_CHUNK = 256;

  /**
   * @brief CONDA's Neighbor order: nearer first, the smaller id on a tie
   * @param a one candidate
   * @param b the other
   * @return whether a sorts before b
   */
  static bool closer(const Neighbor &a, const Neighbor &b);

  static bool closer(const Neighbor &a, const Neighbor &b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
  }

  /** @brief A thread's PQ scratch for condadist, freed at thread exit as a bench starts fresh
   *  worker threads for every run */
  struct CondaScratch {
    uint8_t *bytes = nullptr;
    /** @brief Free the scratch */
    ~CondaScratch();
  };

  CondaScratch::~CondaScratch() {
    // [1]
    if (bytes != nullptr) {
      pipeann::aligned_free(bytes);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condaload(const std::string &base, uint64_t capacity, uint32_t entry) {
    // [1]
    if (capacity < meta_.npoints) {
      LOG(ERROR) << "CONDA capacity " << capacity << " below " << meta_.npoints << " points";
      crash();
    }
    condahead.assign(capacity, 0);
    condalive.assign(capacity, 0);
    stored.assign(capacity, 0);
    condagone.clear();
    condafree.clear();
    condaloc.clear();
#pragma omp parallel for schedule(static, 65536)
    for (int64_t id = 0; id < (int64_t) meta_.npoints; ++id) {
      stored[id] = 1;
      condalive[id] = 1;
      condaloc.insert_or_assign(id2tag((uint32_t) id), (uint32_t) id);
    }
    // [2]
    std::ifstream in(base, std::ios::binary);
    uint32_t npts = 0, dim = 0;
    in.read((char *) &npts, sizeof(uint32_t));
    in.read((char *) &dim, sizeof(uint32_t));
    const uint64_t rows = std::min<uint64_t>(npts, meta_.npoints), block = 65536;
    std::vector<T> chunk(block * dim);
    for (uint64_t st = 0; st < rows; st += block) {
      const uint64_t n = std::min(block, rows - st);
      in.read((char *) chunk.data(), n * dim * sizeof(T));
      for (uint64_t i = 0; i < n; ++i) {
        condahead[st + i] = chunk[i * dim] == std::numeric_limits<T>::max() ? 1 : 0;
      }
    }
    // [3]
    condaep = entry;
    meta_.entry_point = entry;
    meta_.entry_point_id = entry;
    condafresh.store(0);
    LOG(INFO) << "CONDA over " << meta_.npoints << " points, capacity " << capacity
              << ", entry " << entry;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condabatch() {
    // [1]
    condafresh.store(condagone.empty() ? 1 : 0);
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::condaedges(uint32_t id) {
    // [1]
    CondaOp op;
    std::vector<uint32_t> list = condaget(id, op);
    reader->deref(&op.pages);
    return list;
  }

  // Searches read a node's tombstone flag while an insert or a delete sets it, so a single flag goes
  // through relaxed atomics.
  static inline uint8_t headload(const uint8_t *flag) {
    return __atomic_load_n(flag, __ATOMIC_RELAXED);
  }

  static inline void headstore(uint8_t *flag, uint8_t value) {
    __atomic_store_n(flag, value, __ATOMIC_RELAXED);
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::condadead(uint32_t id) {
    // [1]
    return id < condahead.size() && headload(&condahead[id]) != 0;
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::condaout(uint32_t id) {
    // [1]
    return id >= condalive.size() || headload(&condalive[id]) == 0;
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::condareserve() {
    // [1]
    uint32_t id = 0;
    if (!condafree.empty()) {
      id = condafree.back();
      condafree.pop_back();
    } else {
      id = (uint32_t) cur_id++;
    }
    // [2]
    if (id >= condahead.size()) {
      LOG(ERROR) << "CONDA id " << id << " past capacity " << condahead.size();
      crash();
    }
    return id;
  }

  template<typename T, typename TagT>
  const std::vector<uint32_t> &SSDIndex<T, TagT>::condaget(uint32_t id, CondaOp &op) {
    // [1]
    auto it = op.lists.find(id);
    if (it == op.lists.end()) {
      condafetch(std::vector<uint32_t>{id}, op);
      it = op.lists.find(id);
    }
    // [2]
    if (it == op.lists.end()) {
      it = op.lists.insert({id, std::vector<uint32_t>()}).first;
    }
    return it->second;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condafetch(const std::vector<uint32_t> &ids, CondaOp &op) {
    // [1]
    std::vector<uint32_t> want;
    for (uint32_t id : ids) {
      if (id < stored.size() && op.lists.find(id) == op.lists.end()) {
        want.push_back(id);
      }
    }
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    void *ctx = reader->get_ctx();
    for (uint64_t st = 0; st < want.size(); st += CONDA_CHUNK) {
      const uint64_t ed = std::min<uint64_t>(want.size(), st + CONDA_CHUNK);
      std::vector<uint32_t> asked(want.begin() + st, want.begin() + ed);
      // [2]
      auto locked = lock_idx(idx_lock_table, kInvalidID, asked, true);
      std::vector<uint32_t> part;
      for (uint32_t id : asked) {
        if (stored[id]) {
          part.push_back(id);
        }
      }
      const uint64_t bytes = std::max<uint64_t>(part.size(), 1) * io_size;
      char *buf = (char *) reader->alloc_io_buf(bytes, SECTOR_LEN);
      std::vector<IORequest> reqs;
      std::vector<uint32_t> locs;
      for (uint64_t i = 0; i < part.size(); ++i) {
        const uint32_t loc = id2loc(part[i]);
        locs.push_back(loc);
        const uint64_t off = loc_sector_no(loc) * SECTOR_LEN;
        reqs.push_back(IORequest(off, io_size, buf + i * io_size, 0, 0));
      }
      std::vector<uint64_t> refs;
      reader->read_alloc(reqs, ctx, &refs);
      // [3]
      for (uint64_t i = 0; i < part.size(); ++i) {
        DiskNode<T> node = node_from_page(buf + i * io_size, locs[i]);
        op.lists[part[i]].assign(node.nbrs, node.nbrs + node.nnbrs);
        if (op.point.find(part[i]) == op.point.end()) {
          op.point[part[i]].assign(node.coords, node.coords + meta_.data_dim);
        }
      }
      reader->deref(&refs);
      reader->free_io_buf(buf, bytes);
      unlock_idx(idx_lock_table, locked);
    }
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condaedit(uint32_t id, CondaEdit edit, CondaOp &op) {
    // [1]
    if (edit.token == 'A' || edit.token == 'D') {
      condaget(id, op);
    }
    if (edit.token != 'V') {
      auto &list = op.lists[id];
      if (edit.token == 'S') {
        list = edit.whole;
      } else if (edit.token == 'A') {
        list.push_back(edit.entry);
      } else if (edit.token == 'D') {
        list.erase(std::remove(list.begin(), list.end(), edit.entry), list.end());
      } else if (edit.token == 'C') {
        list.clear();
      }
    }
    // [2]
    auto &log = op.edits[id];
    if (log.empty()) {
      op.dirty.push_back(id);
    }
    log.push_back(std::move(edit));
  }

  template<typename T, typename TagT>
  float SSDIndex<T, TagT>::condadist(uint32_t first, uint32_t second, CondaOp &op) {
    // [1]
    if (std::is_floating_point<T>::value) {
      const uint8_t a = headload(&condahead[first]), b = headload(&condahead[second]);
      if (a == 2 || b == 2) {
        return a == b ? 0.0f : std::numeric_limits<float>::infinity();
      }
    }
    // [2]
    if (params.conda == 1) {
      auto a = op.point.find(first), b = op.point.find(second);
      if (a != op.point.end() && b != op.point.end()) {
        return dist_cmp->compare(a->second.data(), b->second.data(), (uint32_t) meta_.data_dim);
      }
    }
    // [3]
    static thread_local CondaScratch scratch;
    if (scratch.bytes == nullptr) {
      const uint64_t bytes = 4 * AbstractNeighbor<T>::MAX_BYTES_PER_NBR;
      pipeann::alloc_aligned((void **) &scratch.bytes, bytes, 256);
    }
    float out = 0.0f;
    nbr_handler->compute_dists(first, &second, 1, &out, scratch.bytes);
    op.pairs++;
    return out;
  }

  template<typename T, typename TagT>
  std::vector<uint32_t> SSDIndex<T, TagT>::condaclose(const std::vector<uint32_t> &pool,
                                                      uint32_t self, CondaOp &op) {
    // [1]
    std::vector<Neighbor> cands;
    for (uint32_t t : pool) {
      if (t != self) {
        cands.push_back(Neighbor(t, condadist(t, self, op), true));
      }
    }
    // [2]
    const uint64_t k = std::min<uint64_t>(CONDA_LINKS, cands.size());
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(), closer);
    std::vector<uint32_t> ret;
    for (uint64_t i = 0; i < k; ++i) {
      ret.push_back(cands[i].id);
    }
    return ret;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condasearch(uint32_t self, CondaSeek seek, uint32_t listn, CondaOp &op,
                                      std::vector<Neighbor> &expanded, QueryStats *stats) {
    // [1] the walk starts from a random live vertex and its list
    const uint32_t start = randentry();
    const std::vector<uint32_t> nbrs = condaget(start, op);
    tsl::robin_set<uint32_t> seen, pushed;
    std::vector<uint32_t> init, seeds;
    if (seek != CondaSeek::HEAL) {
      init.push_back(start);
    }
    for (uint64_t i = 0; i < nbrs.size() && i + 1 < listn; ++i) {
      const uint32_t id = nbrs[i];
      if (seek != CondaSeek::HEAL && id == self) {
        continue;
      }
      if (seek == CondaSeek::LAZY && condadead(id)) {
        condaedit(start, CondaEdit{'D', id, {}}, op);
        seen.insert(id);
        continue;
      }
      init.push_back(id);
    }
    for (uint32_t id : init) {
      if (id != self && seen.find(id) == seen.end() && pushed.insert(id).second) {
        seeds.push_back(id);
      }
    }
    // [2]
    uint32_t cur = kInvalidID;
    std::vector<uint32_t> order;
    const bool tomb = params.conda == 1 && std::is_floating_point<T>::value;
    auto approx = [&](unsigned id) -> bool {
      if (id == self || seen.find(id) != seen.end()) {
        return false;
      }
      if (seek == CondaSeek::LAZY && condadead(id)) {
        seen.insert(id);
        condaedit(cur, CondaEdit{'D', (uint32_t) id, {}}, op);
        return false;
      }
      return !(tomb && headload(&condahead[id]) == 2);
    };
    auto verify = [&](unsigned id, const DiskNode<T> &node) -> bool {
      cur = id;
      order.push_back(id);
      if (op.lists.find(id) == op.lists.end()) {
        op.lists[id].assign(node.nbrs, node.nbrs + node.nnbrs);
      }
      if (op.point.find(id) == op.point.end()) {
        op.point[id].assign(node.coords, node.coords + meta_.data_dim);
      }
      return true;
    };
    InsertContext ctx(kExpandedNodesFactor * listn, aligned_dim);
    std::vector<Neighbor> full;
    const std::vector<T> query = op.point[self];
    if (params.conda == 2) {
      this->beam_search_common(query.data(), 0, listn, 1, approx, verify, full, stats, &ctx,
                               &seeds);
    } else {
      const float unbounded = std::numeric_limits<float>::infinity();
      const uint32_t width = seek == CondaSeek::HEAL ? params.delete_beam_width : params.beam_width;
      this->pipe_search_common(query.data(), 0, listn, listn, width, io_size, false,
                               approx, verify, full, stats, &ctx, unbounded, &seeds);
    }
    // [3]
    expanded.clear();
    for (uint32_t id : order) {
      expanded.push_back(Neighbor(id, condadist(id, self, op), true));
    }
    op.hints.insert(op.hints.end(), ctx.hint_pages.begin(), ctx.hint_pages.end());
    op.pages.insert(op.pages.end(), ctx.page_ref.begin(), ctx.page_ref.end());
  }

  template<typename T, typename TagT>
  uint32_t SSDIndex<T, TagT>::condaselect(uint32_t self, const std::vector<Neighbor> &expanded,
                                          CondaOp &op, std::vector<uint32_t> &pruned,
                                          std::vector<uint32_t> &result) {
    // [1]
    op.turns++;
    const uint64_t n = expanded.size();
    std::vector<float> occlude(n, std::numeric_limits<float>::max());
    std::vector<uint32_t> occluder(n, 0);
    pruned.assign(n, std::numeric_limits<uint32_t>::max());
    result.clear();
    uint32_t maxid = 0;
    for (uint64_t i = 0; i < n && result.size() < params.R; ++i) {
      const Neighbor p = expanded[i];
      // [2]
      if (occlude[i] == std::numeric_limits<float>::max()) {
        result.push_back(p.id);
        maxid = (uint32_t) i;
      } else if (occlude[i] < p.distance) {
        pruned[i] = occluder[i];
        continue;
      } else {
        result.push_back(p.id);
        maxid = (uint32_t) i;
        condaedit(occluder[i], CondaEdit{'D', p.id, {}}, op);
        condaedit(occluder[i], CondaEdit{'A', self, {}}, op);
      }
      // [3]
      const std::vector<uint32_t> list = condaget(p.id, op);
      tsl::robin_set<uint32_t> nset(list.begin(), list.end());
      for (uint64_t j = i + 1; j < n; ++j) {
        const Neighbor &q = expanded[j];
        if (q.distance > occlude[j] || nset.find(q.id) == nset.end()) {
          continue;
        }
        const float d = condadist(p.id, q.id, op);
        if (d < occlude[j]) {
          occlude[j] = d;
          occluder[j] = p.id;
        }
      }
    }
    return maxid;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condalink(uint32_t self, const std::vector<Neighbor> &expanded,
                                    std::vector<uint32_t> &pruned, uint32_t maxid, CondaOp &op) {
    // [1]
    if (expanded.empty()) {
      return;
    }
    const uint64_t limit = expanded.size() / 4;
    for (uint64_t i = 0; i <= maxid; ++i) {
      const Neighbor nbr = expanded[i];
      const uint32_t des = nbr.id;
      if (des == self || pruned[i] != std::numeric_limits<uint32_t>::max()) {
        continue;
      }
      const std::vector<uint32_t> list = condaget(des, op);
      if (std::find(list.begin(), list.end(), self) != list.end()) {
        continue;
      }
      // [2]
      if (list.size() < params.R) {
        condaedit(des, CondaEdit{'A', self, {}}, op);
        continue;
      }
      // [3]
      std::vector<Neighbor> pool;
      for (uint32_t id : list) {
        pool.push_back(Neighbor(id, condadist(des, id, op), true));
      }
      pool.push_back(Neighbor(self, nbr.distance, true));
      std::vector<uint32_t> kept;
      const bool connect = condaalpha(des, pool, self, kept, op);
      condaedit(des, CondaEdit{'S', 0, kept}, op);
      if (!connect) {
        for (uint64_t j = i + 1; j < limit; ++j) {
          if (pruned[j] == des) {
            pruned[j] = std::numeric_limits<uint32_t>::max();
          }
        }
      }
    }
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::condaalpha(uint32_t center, std::vector<Neighbor> &pool, uint32_t self,
                                     std::vector<uint32_t> &result, CondaOp &op) {
    // [1]
    op.turns++;
    std::sort(pool.begin(), pool.end(), closer);
    if (pool.size() > params.C) {
      pool.resize(params.C);
    }
    // [2]
    result.clear();
    bool connect = false;
    for (uint64_t i = 0; i < pool.size() && result.size() < params.R; ++i) {
      bool occlude = false;
      for (uint32_t r : result) {
        if (params.alpha * condadist(r, pool[i].id, op) < pool[i].distance) {
          occlude = true;
        }
      }
      if (!occlude) {
        result.push_back(pool[i].id);
        connect = connect || pool[i].id == self;
      }
    }
    return connect;
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condacrng(uint32_t center, std::vector<Neighbor> &pool,
                                    std::vector<uint32_t> &result, CondaOp &op) {
    // [1]
    std::vector<uint32_t> ids;
    for (auto &p : pool) {
      ids.push_back(p.id);
    }
    condafetch(ids, op);
    std::sort(pool.begin(), pool.end(), closer);
    const uint64_t n = pool.size();
    std::vector<float> occlude(n, std::numeric_limits<float>::max());
    std::vector<uint32_t> occluder(n, std::numeric_limits<uint32_t>::max());
    result.clear();
    for (uint64_t i = 0; i < n && result.size() < params.R; ++i) {
      const Neighbor p = pool[i];
      // [2]
      if (occlude[i] != std::numeric_limits<float>::max()) {
        if (occlude[i] < p.distance) {
          continue;
        }
        condaedit(occluder[i], CondaEdit{'D', p.id, {}}, op);
      }
      result.push_back(p.id);
      // [3]
      const std::vector<uint32_t> list = condaget(p.id, op);
      tsl::robin_set<uint32_t> nset(list.begin(), list.end());
      for (uint64_t j = i + 1; j < n; ++j) {
        const Neighbor &q = pool[j];
        if (q.distance > occlude[j] || nset.find(q.id) == nset.end()) {
          continue;
        }
        const float d = condadist(p.id, q.id, op);
        if (d < occlude[j]) {
          occlude[j] = d;
          occluder[j] = p.id;
        }
      }
    }
  }

  template<typename T, typename TagT>
  int SSDIndex<T, TagT>::condainsert(const T *point, const TagT &tag, QueryStats *stats,
                                     uint64_t *pages, uint64_t *close, uint64_t begun) {
    auto run = [&](CondaSeek seek) -> int {
      // [1]
      CondaOp op;
      uint32_t id = 0;
      {
        std::lock_guard<std::mutex> guard(condamu);
        id = condareserve();
        condaloc.insert_or_assign(tag, id);
        condalive[id] = 1;
      }
      tags.insert_or_assign(id, tag);
      // [2]
      op.point[id].assign(point, point + meta_.data_dim);
      nbr_handler->insert(op.point[id].data(), id);
      headstore(&condahead[id], point[0] == std::numeric_limits<T>::max() ? 1 : 0);
      condaedit(id, CondaEdit{'V', 0, {}}, op);
      // [3]
      std::vector<Neighbor> expanded;
      condasearch(id, seek, params.L, op, expanded, stats);
      std::sort(expanded.begin(), expanded.end(), closer);
      std::vector<uint32_t> pruned, result, own;
      const uint32_t maxid = condaselect(id, expanded, op, pruned, result);
      for (uint32_t r : result) {
        if (r != id) {
          own.push_back(r);
        }
      }
      condaedit(id, CondaEdit{'S', 0, own}, op);
      // [4]
      condalink(id, expanded, pruned, maxid, op);
      // the background writer sets pages and close once the flush's pages are written, as for an OdinANN insert
      const uint64_t written = condaflush(op, pages, close, begun);
      reader->deref(&op.pages);
      if (stats != nullptr) {
        stats->n_out_edges = own.size();
        stats->n_pages_touched = written;
        stats->n_passes += op.turns;
        stats->n_sdc += op.pairs;
      }
      return (int) id;
    };
    // [5] the batch's first insert walks FULL alone; one that finds it taken never queues for the writer lock,
    // which readers would starve, but takes the reader lock that waits for the FULL insert to finish
    while (condafresh.load() == 1) {
      std::unique_lock<std::shared_mutex> first(condagate, std::try_to_lock);
      if (!first.owns_lock()) {
        std::this_thread::yield();
        continue;
      }
      if (condafresh.load() == 1) {
        condafresh.store(2);
        const int id = run(CondaSeek::FULL);
        condafresh.store(0);
        return id;
      }
    }
    std::shared_lock<std::shared_mutex> rest(condagate);
    return run(CondaSeek::LAZY);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condarepair(uint32_t id, bool tomb) {
    // [1]
    CondaOp op;
    condafetch(std::vector<uint32_t>{id}, op);
    std::vector<Neighbor> expanded;
    condasearch(id, CondaSeek::HEAL, CONDA_LIST, op, expanded, nullptr);
    // [2]
    std::vector<Neighbor> sorted = expanded;
    std::sort(sorted.begin(), sorted.end(), closer);
    std::vector<uint32_t> pool;
    for (uint64_t i = 0; i < CONDA_POOL && i < sorted.size(); ++i) {
      if (!condaout(sorted[i].id)) {
        pool.push_back(sorted[i].id);
      }
    }
    // [3]
    std::unordered_map<unsigned, std::vector<unsigned>> updates;
    std::vector<uint32_t> in;
    for (auto &node : expanded) {
      if (condaout(node.id)) {
        continue;
      }
      const std::vector<uint32_t> &list = condaget(node.id, op);
      if (std::find(list.begin(), list.end(), id) != list.end()) {
        in.push_back(node.id);
      }
    }
    for (uint32_t u : in) {
      std::vector<unsigned> list = condaget(u, op);
      tsl::robin_set<unsigned> merged(list.begin(), list.end());
      for (uint32_t c : condaclose(pool, u, op)) {
        merged.insert(c);
      }
      list.assign(merged.begin(), merged.end());
      updates[u] = std::move(list);
    }
    // [4]
    const std::vector<uint32_t> out = condaget(id, op);
    for (uint32_t w : out) {
      if (condaout(w)) {
        continue;
      }
      for (uint32_t y : condaclose(pool, w, op)) {
        std::vector<unsigned> list;
        if (updates[y].empty()) {
          const std::vector<uint32_t> &held = condaget(y, op);
          list.assign(held.begin(), held.end());
        } else {
          list = updates[y];
        }
        tsl::robin_set<unsigned> merged(list.begin(), list.end());
        merged.insert(w);
        list.assign(merged.begin(), merged.end());
        updates[y] = std::move(list);
      }
    }
    // [5] each over-full patch's candidates by distance to it, then the lists they prune by in one batch
    std::vector<std::pair<uint32_t, std::vector<uint32_t>>> patches;
    std::vector<std::vector<Neighbor>> pools;
    std::vector<uint32_t> wide;
    for (auto &kv : updates) {
      std::vector<uint32_t> filtered;
      for (unsigned x : kv.second) {
        if (x != kv.first && !condaout(x)) {
          filtered.push_back(x);
        }
      }
      std::vector<Neighbor> cands;
      if (filtered.size() > params.R) {
        for (uint32_t x : filtered) {
          cands.push_back(Neighbor(x, condadist(kv.first, x, op), true));
        }
        wide.insert(wide.end(), filtered.begin(), filtered.end());
      }
      patches.emplace_back(kv.first, std::move(filtered));
      pools.push_back(std::move(cands));
    }
    condafetch(wide, op);
    // [6]
    for (uint64_t k = 0; k < patches.size(); ++k) {
      const auto &[center, filtered] = patches[k];
      if (filtered.size() > params.R) {
        std::vector<uint32_t> kept;
        condacrng(center, pools[k], kept, op);
        condaedit(center, CondaEdit{'S', 0, kept}, op);
      } else {
        condaedit(center, CondaEdit{'S', 0, filtered}, op);
      }
    }
    // [7]
    if (tomb) {
      condatomb(id, op);
    }
    condaflush(op);
    reader->deref(&op.pages);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condaclear(uint32_t id) {
    // [1]
    CondaOp op;
    condatomb(id, op);
    // [2]
    condaflush(op);
    reader->deref(&op.pages);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condatomb(uint32_t id, CondaOp &op) {
    // [1]
    condaedit(id, CondaEdit{'C', 0, {}}, op);
    op.point[id].assign(meta_.data_dim, std::numeric_limits<T>::max());
    condaedit(id, CondaEdit{'V', 0, {}}, op);
    nbr_handler->insert(op.point[id].data(), id);
    headstore(&condahead[id], 2);
  }

  template<typename T, typename TagT>
  void SSDIndex<T, TagT>::condadelete(const std::vector<TagT> &tags, uint32_t nthreads) {
    // [1]
    nthreads = std::max<uint32_t>(nthreads, 1);
    for (const TagT &tag : tags) {
      uint32_t id = kInvalidID;
      if (!condaloc.find(tag, id)) {
        LOG(ERROR) << "CONDA tag " << tag << " not found";
        crash();
      }
      condaloc.erase(tag);
      condalive[id] = 0;
      condagone.insert(id);
    }
    // [2]
    const std::vector<uint32_t> gone(condagone.begin(), condagone.end());
#pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t) gone.size(); ++i) {
      condarepair(gone[i]);
    }
    // [3]
    if (condagone.find(condaep) != condagone.end()) {
      uint32_t entry = condaep;
      for (uint64_t id = 0; id < condalive.size(); ++id) {
        if (condalive[id]) {
          entry = (uint32_t) id;
          break;
        }
      }
      std::lock_guard<WriterFirstMutex> swap(merge_lock);
      condaep = entry;
      meta_.entry_point = entry;
      meta_.entry_point_id = entry;
    }
    // [4]
#pragma omp parallel for schedule(dynamic, 1) num_threads(nthreads)
    for (int64_t i = 0; i < (int64_t) gone.size(); ++i) {
      condaclear(gone[i]);
      std::lock_guard<std::mutex> guard(condamu);
      condafree.push_back(gone[i]);
    }
    condagone.clear();
  }

  template<typename T, typename TagT>
  bool SSDIndex<T, TagT>::condaremove(TagT tag) {
    // [1] MarkDelete: the tag leaves the live map, its vertex the live set
    std::shared_lock<std::shared_mutex> gate(condagate);
    uint32_t id = kInvalidID;
    {
      std::lock_guard<std::mutex> guard(condamu);
      if (!condaloc.find(tag, id)) {
        return false;
      }
      condaloc.erase(tag);
      headstore(&condalive[id], 0);
    }
    // [2] a deleted entry point gives way to the first live vertex, as in condadelete, before its tombstone lands
    if (id == condaep) {
      uint32_t entry = condaep;
      for (uint64_t at = 0; at < condalive.size(); ++at) {
        if (!condaout((uint32_t) at)) {
          entry = (uint32_t) at;
          break;
        }
      }
      std::lock_guard<WriterFirstMutex> swap(merge_lock);
      condaep = entry;
      meta_.entry_point = entry;
      meta_.entry_point_id = entry;
    }
    // [3] relink its in-neighbors and out-neighbors and write its tombstone, in one flush
    condarepair(id, true);
    // [4] the tombstone on the free stack, for a later insert to reuse
    std::lock_guard<std::mutex> guard(condamu);
    condafree.push_back(id);
    return true;
  }

  template<typename T, typename TagT>
  uint64_t SSDIndex<T, TagT>::condaflush(CondaOp &op, uint64_t *pages, uint64_t *close, uint64_t begun) {
    // [1]
    const std::vector<uint32_t> ids = op.dirty;
    const uint64_t n = ids.size();
    if (n == 0) {
      if (pages != nullptr) {
        *pages = 0;
      }
      if (close != nullptr) {
        *close = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count() - begun;
      }
      return 0;
    }
    if (meta_.nnodes_per_sector == 0) {
      LOG(ERROR) << "CONDA needs records that share pages";
      crash();
    }
    uint8_t token = condaslots.pop();
    while (token == 0) {
      condaslots.wait_for_push_notify();
      token = condaslots.pop();
    }
    void *ctx = reader->get_ctx();
    // [2]
    std::set<uint64_t> hinted;
    std::vector<uint64_t> locs = alloc_loc((int) n, op.hints, hinted);
    std::map<uint64_t, uint32_t> owned;
    for (uint64_t loc : locs) {
      owned[loc_sector_no(loc)]++;
    }
    std::vector<IORequest> plock;
    for (auto &kv : owned) {
      plock.push_back(IORequest(kv.first * SECTOR_LEN, io_size, nullptr, 0, 0));
    }
    auto plocked = pipeann::lockReqs(page_lock_table, plock);
    lock_vec(vec_lock_table, kInvalidID, ids);
    // [3] a live record whose list an edit replaces and whose vector op holds is rebuilt without its old page;
    // a dead one is read, for a tombstone written meanwhile keeps its vector
    std::vector<uint8_t> needold(n, 0);
    for (uint64_t i = 0; i < n; ++i) {
      bool list = false, vec = false;
      for (auto &e : op.edits.find(ids[i])->second) {
        list = list || e.token == 'S' || e.token == 'C';
        vec = vec || e.token == 'V';
      }
      vec = vec || (op.point.find(ids[i]) != op.point.end() && !condadead(ids[i]));
      needold[i] = stored[ids[i]] && !(list && vec);
    }
    std::set<uint64_t> toread;
    for (auto &kv : owned) {
      for (uint32_t held : get_page_layout((uint32_t) kv.first)) {
        if (held != kInvalidID && held != kAllocatedID) {
          toread.insert(kv.first);
        }
      }
    }
    std::vector<uint64_t> olds(n, kInvalidID);
    for (uint64_t i = 0; i < n; ++i) {
      if (stored[ids[i]]) {
        olds[i] = id2loc(ids[i]);
      }
      if (needold[i]) {
        toread.insert(loc_sector_no(olds[i]));
      }
    }
    std::set<uint64_t> all(toread.begin(), toread.end());
    for (auto &kv : owned) {
      all.insert(kv.first);
    }
    char *buf = (char *) reader->alloc_io_buf(all.size() * io_size, SECTOR_LEN);
    memset(buf, 0, all.size() * io_size);
    std::map<uint64_t, char *> slot;
    uint64_t k = 0;
    for (uint64_t page : all) {
      slot[page] = buf + (k++) * io_size;
    }
    std::vector<IORequest> reads;
    for (uint64_t page : toread) {
      reads.push_back(IORequest(page * SECTOR_LEN, io_size, slot[page], 0, 0));
    }
    std::vector<uint64_t> rrefs;
    for (uint64_t st = 0; st < reads.size(); st += CONDA_CHUNK) {
      const uint64_t ed = std::min<uint64_t>(reads.size(), st + CONDA_CHUNK);
      std::vector<IORequest> part(reads.begin() + st, reads.begin() + ed);
      reader->read_alloc(part, ctx, &rrefs);
    }
    // [4]
    for (uint64_t i = 0; i < n; ++i) {
      const uint32_t id = ids[i];
      std::vector<uint32_t> list;
      std::vector<T> coords(meta_.data_dim, 0);
      if (needold[i]) {
        DiskNode<T> old = node_from_page(slot[loc_sector_no(olds[i])], olds[i]);
        list.assign(old.nbrs, old.nbrs + old.nnbrs);
        coords.assign(old.coords, old.coords + meta_.data_dim);
      } else if (op.point.find(id) != op.point.end()) {
        coords = op.point.find(id)->second;
      }
      for (auto &e : op.edits.find(id)->second) {
        if (e.token == 'S') {
          list = e.whole;
        } else if (e.token == 'A') {
          list.push_back(e.entry);
        } else if (e.token == 'D') {
          list.erase(std::remove(list.begin(), list.end(), e.entry), list.end());
        } else if (e.token == 'C') {
          list.clear();
        } else if (e.token == 'V') {
          coords = op.point.find(id)->second;
        }
      }
      if (list.size() > meta_.range) {
        LOG(WARNING) << "CONDA list of " << id << " holds " << list.size() << " past "
                     << meta_.range << ", alpha-pruned";
        std::vector<Neighbor> pool;
        for (uint32_t x : list) {
          pool.push_back(Neighbor(x, condadist(id, x, op), true));
        }
        condaalpha(id, pool, kInvalidID, list, op);
      }
      DiskNode<T> dst = node_from_page(slot[loc_sector_no(locs[i])], locs[i]);
      memset((char *) dst.coords, 0, meta_.max_node_len);
      memcpy(dst.coords, coords.data(), meta_.data_dim * sizeof(T));
      dst.nnbrs = (uint16_t) list.size();
      dst.n_dense_nbrs = 0;
      memcpy(dst.nbrs, list.data(), list.size() * sizeof(uint32_t));
    }
    // [5] the written pages' 4k writes, runs of consecutive pages merged into one continuous write as an
    // OdinANN insert's; owned is ordered and consecutive pages sit next to each other in buf
    std::vector<IORequest> writes;
    uint64_t run = 0, last = 0;
    for (auto &kv : owned) {
      const uint64_t off = kv.first * SECTOR_LEN;
      if (writes.empty() || off != last + io_size) {
        if (!writes.empty()) {
          writes.back().len = run * io_size;
        }
        run = 0;
        writes.push_back(IORequest(off, io_size, slot[kv.first], 0, 0));
      }
      ++run;
      last = off;
    }
    if (!writes.empty()) {
      writes.back().len = run * io_size;
    }
    std::vector<uint64_t> wrefs;
    reader->wbc_write(writes, ctx, &wrefs);
    auto ilocked = lock_idx(idx_lock_table, kInvalidID, ids);
    std::vector<uint64_t> freed;
    for (uint64_t i = 0; i < n; ++i) {
      if (olds[i] != kInvalidID) {
        freed.push_back(olds[i]);
      }
      set_id2loc(ids[i], (uint32_t) locs[i]);
      stored[ids[i]] = 1;
    }
    erase_and_set_loc(freed, locs, ids);
    unlock_idx(idx_lock_table, ilocked);
    // [6] the background writer writes the pages, then releases their locks, references, buffer and the
    // queue slot [1] took, as an OdinANN insert's; meanwhile reads find the pages' new contents in the page
    // cache wbc_write filled
    unlock_vec(vec_lock_table, kInvalidID, ids);
    reader->deref(&rrefs);
    const uint64_t written = owned.size();
    bg_tasks.push(new BgTask{.thread_data = nullptr,
                             .writes = std::move(writes),
                             .pages_to_unlock = std::move(plocked),
                             .pages_to_deref = std::move(wrefs),
                             .pages = pages,
                             .close = close,
                             .begun = begun,
                             .buf = buf,
                             .buflen = all.size() * io_size});
    bg_tasks.push_notify_all();
    return written;
  }

  template class SSDIndex<float>;
  template class SSDIndex<int8_t>;
  template class SSDIndex<uint8_t>;
}  // namespace pipeann
