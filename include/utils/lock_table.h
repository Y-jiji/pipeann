#ifndef LOCK_TABLE_H_
#define LOCK_TABLE_H_
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <omp.h>
#include <shared_mutex>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "utils/libcuckoo/cuckoohash_map.hh"
#include "utils/log.h"

inline void thread_pause() {
  // Use pause instruction to reduce contention in tight loops.
#ifdef __x86_64__
  asm volatile("pause" ::: "memory");
#endif
}

namespace pipeann {
  /** @brief Read-only polls of a taken latch before a waiter sleeps on it */
  constexpr int LATCH_SPINS = 256;

  /**
   * @brief Reader-writer latch with no owner, so any thread may release it, as the background IO
   *        thread releases the page locks insert threads took; state is 0 when free, -1 when
   *        write-held and n > 0 when read-held by n, sleep counts the threads asleep on state
   */
  struct SparseLatch {
    std::atomic<int32_t> state{0};
    std::atomic<int32_t> sleep{0};

    /**
     * @brief Take a read hold unless the latch is write-held
     * @return whether the hold was taken
     */
    bool tryrd();

    /**
     * @brief Take the write hold if the latch is free
     * @return whether the hold was taken
     */
    bool trywr();

    /** @brief Take a read hold, waiting while the latch is write-held */
    void rd();

    /** @brief Take the write hold, waiting while the latch is held */
    void wr();

    /**
     * @brief Return once the latch looked takeable, after a read-only spin or a futex sleep
     * @param reader whether a read hold is wanted
     */
    void wait(bool reader);

    /** @brief Drop one hold, the write hold or a read hold, and wake the sleepers if any */
    void release();
  };

  inline bool SparseLatch::tryrd() {
    int32_t s = state.load(std::memory_order_relaxed);
    while (s >= 0) {
      if (state.compare_exchange_weak(s, s + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
        return true;
      }
    }
    return false;
  }

  inline bool SparseLatch::trywr() {
    int32_t s = 0;
    return state.compare_exchange_strong(s, -1, std::memory_order_acquire, std::memory_order_relaxed);
  }

  inline void SparseLatch::rd() {
    while (!tryrd()) {
      wait(true);
    }
  }

  inline void SparseLatch::wr() {
    while (!trywr()) {
      wait(false);
    }
  }

  inline void SparseLatch::wait(bool reader) {
    // [1]
    const auto open = [reader](int32_t s) { return reader ? s >= 0 : s == 0; };
    for (int spin = 0; spin < LATCH_SPINS; ++spin) {
      if (open(state.load(std::memory_order_relaxed))) {
        return;
      }
      thread_pause();
    }
    // [2]
    sleep.fetch_add(1, std::memory_order_seq_cst);
    const int32_t s = state.load(std::memory_order_seq_cst);
    if (!open(s)) {
      syscall(SYS_futex, reinterpret_cast<int32_t *>(&state), FUTEX_WAIT_PRIVATE, s, nullptr, nullptr, 0);
    }
    sleep.fetch_sub(1, std::memory_order_relaxed);
  }

  inline void SparseLatch::release() {
    if (state.load(std::memory_order_relaxed) == -1) {
      state.store(0, std::memory_order_seq_cst);
    } else {
      state.fetch_sub(1, std::memory_order_seq_cst);
    }
    if (sleep.load(std::memory_order_seq_cst) > 0) {
      syscall(SYS_futex, reinterpret_cast<int32_t *>(&state), FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr,
              0);
    }
  }

  /**
   * @brief A lock per key, made on the key's first use and erased once no thread holds or waits on
   *        it; a thread counts itself on the key's entry with one map update, then waits on the
   *        entry's latch outside the map, so no retry loop holds the map's bucket spinlocks against
   *        the release of the key it waits for
   * @tparam KEY key type
   * @tparam HASH hash of a key
   */
  template<class KEY, class HASH = std::hash<KEY>>
  class SparseLockTable {
   public:
    /** @brief An empty table */
    SparseLockTable();

    /**
     * @brief Take a read hold on key unless it is write-held
     * @param key the key
     * @return 0 when taken, EBUSY when not
     */
    int tryrdlock(const KEY &key);

    /**
     * @brief Take the write hold on key if nobody holds it
     * @param key the key
     * @return 0 when taken, EBUSY when not
     */
    int trywrlock(const KEY &key);

    /**
     * @brief Take a read hold on key, waiting on its latch while it is write-held
     * @param key the key
     */
    void rdlock(const KEY &key);

    /**
     * @brief Take the write hold on key, waiting on its latch while it is held
     * @param key the key
     */
    void wrlock(const KEY &key);

    /**
     * @brief Drop a hold on key, from any thread
     * @param key the key
     */
    void unlock(const KEY &key);

    /**
     * @brief Keys held or waited on
     * @return the number of entries
     */
    size_t size();

   private:
    /**
     * @brief Count one more holder or waiter on the key's entry, making the entry if absent
     * @param key the key
     * @return the entry's latch
     */
    SparseLatch *pin(const KEY &key);

    /**
     * @brief Count one less holder or waiter on the key's entry, erasing it once unused
     * @param key the key
     * @param held whether to drop a hold on the entry's latch first
     */
    void unpin(const KEY &key, bool held);

    libcuckoo::cuckoohash_map<KEY, std::pair<SparseLatch *, int>, HASH> *table;
  };

  template<class KEY, class HASH>
  SparseLockTable<KEY, HASH>::SparseLockTable()
      : table(new libcuckoo::cuckoohash_map<KEY, std::pair<SparseLatch *, int>, HASH>()) {
  }

  template<class KEY, class HASH>
  int SparseLockTable<KEY, HASH>::tryrdlock(const KEY &key) {
    if (pin(key)->tryrd()) {
      return 0;
    }
    unpin(key, false);
    return EBUSY;
  }

  template<class KEY, class HASH>
  int SparseLockTable<KEY, HASH>::trywrlock(const KEY &key) {
    if (pin(key)->trywr()) {
      return 0;
    }
    unpin(key, false);
    return EBUSY;
  }

  template<class KEY, class HASH>
  void SparseLockTable<KEY, HASH>::rdlock(const KEY &key) {
    pin(key)->rd();
  }

  template<class KEY, class HASH>
  void SparseLockTable<KEY, HASH>::wrlock(const KEY &key) {
    pin(key)->wr();
  }

  template<class KEY, class HASH>
  void SparseLockTable<KEY, HASH>::unlock(const KEY &key) {
    unpin(key, true);
  }

  template<class KEY, class HASH>
  size_t SparseLockTable<KEY, HASH>::size() {
    return table->size();
  }

  template<class KEY, class HASH>
  SparseLatch *SparseLockTable<KEY, HASH>::pin(const KEY &key) {
    SparseLatch *latch = nullptr;
    table->upsert(key, [&](std::pair<SparseLatch *, int> &entry, libcuckoo::UpsertContext ctx) {
      if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
        entry = std::make_pair(new SparseLatch, 0);
      }
      entry.second++;
      latch = entry.first;
    });
    return latch;
  }

  template<class KEY, class HASH>
  void SparseLockTable<KEY, HASH>::unpin(const KEY &key, bool held) {
    table->erase_fn(key, [&](std::pair<SparseLatch *, int> &entry) {
      if (entry.second == 0) {
        LOG(ERROR) << "SparseLockTable: unlock a non-locked key: " << key;
        __builtin_trap();
      }
      if (held) {
        entry.first->release();
      }
      entry.second--;
      if (entry.second == 0) {
        delete entry.first;
      }
      return entry.second == 0;
    });
  }

  template<class K, class HashFunction = std::hash<K>>
  class SparseReadLockGuard {
   public:
    SparseReadLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->rdlock(key_);
    }

    ~SparseReadLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  template<class K, class HashFunction = std::hash<K>>
  class SparseWriteLockGuard {
   public:
    SparseWriteLockGuard(SparseLockTable<K, HashFunction> *table, const K &key) : table_(table), key_(key) {
      table_->wrlock(key_);
    }

    ~SparseWriteLockGuard() {
      table_->unlock(key_);
    }

   private:
    SparseLockTable<K, HashFunction> *table_;
    K key_;
  };

  class LockTable {
   public:
    LockTable(size_t size) : size_(size) {
      locks_ = new pthread_rwlock_t[size];
      for (size_t i = 0; i < size; i++) {
        pthread_rwlock_init(&locks_[i], nullptr);
      }
    }
    ~LockTable() {
      for (size_t i = 0; i < size_; i++) {
        pthread_rwlock_destroy(&locks_[i]);
      }
      delete[] locks_;
    }

    inline pthread_rwlock_t *rdlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_rdlock(lock);
      return lock;
    }

    inline uint64_t pos(uint64_t key) {
      return Hash(key) % size_;
    }

    inline pthread_rwlock_t *wrlock(uint32_t key) {
      auto lock = &locks_[Hash(key) % size_];
      pthread_rwlock_wrlock(lock);
      return lock;
    }

    inline bool tryrdlock(uint32_t key) {
      return (pthread_rwlock_tryrdlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline bool trywrlock(uint32_t key) {
      return (pthread_rwlock_trywrlock(&locks_[Hash(key) % size_]) == 0);
    }

    inline void unlock(pthread_rwlock_t *lock) {
      pthread_rwlock_unlock(lock);
    }

    inline void unlock(uint32_t key) {
      pthread_rwlock_unlock(&locks_[Hash(key) % size_]);
    }

   private:
    size_t size_;
    pthread_rwlock_t *locks_;

    static const uint32_t c1 = 0xcc9e2d51;
    static const uint32_t c2 = 0x1b873593;

    static uint32_t fmix(uint32_t h) {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;
      return h;
    }

    static uint32_t Rotate32(uint32_t val, int shift) {
      // Avoid shifting by 32: doing so yields an undefined result.
      return shift == 0 ? val : ((val >> shift) | (val << (32 - shift)));
    }

    static uint32_t Mur(uint32_t a, uint32_t h) {
      // Helper from Murmur3 for combining two 32-bit values.
      a *= c1;
      a = Rotate32(a, 17);
      a *= c2;
      h ^= a;
      h = Rotate32(h, 19);
      return h * 5 + 0xe6546b64;
    }

    static uint32_t Hash32Len0to4(const char *s, size_t len) {
      uint32_t b = 0;
      uint32_t c = 9;
      for (size_t i = 0; i < len; i++) {
        signed char v = static_cast<signed char>(s[i]);
        b = b * c1 + static_cast<uint32_t>(v);
        c ^= b;
      }
      return fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
    }

    static uint32_t Hash(uint32_t x) {
      return Hash32Len0to4((const char *) &x, sizeof(uint32_t));
    }
  };

  // RAII to avoid forgetting to unlock.
  class LockGuard {
   public:
    LockGuard(pthread_rwlock_t *lock) : lock_(lock) {
    }
    LockGuard &operator=(const LockGuard &) = delete;
    LockGuard &operator=(LockGuard &&rhs) {
      lock_ = rhs.lock_;
      rhs.lock_ = nullptr;
      return *this;
    }
    ~LockGuard() {
      if (lock_) {
        pthread_rwlock_unlock(lock_);
        lock_ = nullptr;
      }
    }

   private:
    pthread_rwlock_t *lock_;
  };

  // Used by fixed-size arrays (with resize).
  // Why use this? Resize is very rare, but a single shared_mutex incurs cache ping-pong.
  struct ReaderOptSharedMutex {
    static constexpr int N = 128;
    struct alignas(128) CachelineAlignedMutex {
      std::shared_mutex mutex;
      char padding[128 - sizeof(std::shared_mutex)];
    } locks_[N];
    void lock_shared(size_t idx = omp_get_thread_num()) {
      locks_[idx % N].mutex.lock_shared();
    }
    void lock() {
      for (size_t i = 0; i < N; ++i) {
        locks_[i].mutex.lock();
      }
    }
    void unlock_shared(size_t idx = omp_get_thread_num()) {
      locks_[idx % N].mutex.unlock_shared();
    }
    void unlock() {
      for (size_t i = 0; i < N; ++i) {
        locks_[i].mutex.unlock();
      }
    }
  };

}  // namespace pipeann

#endif  // LOCK_TABLE_H_
