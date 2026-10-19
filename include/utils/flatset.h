#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace pipeann {
  /** @brief log2 of a FlatSet's initial slots */
  constexpr size_t FLAT_BITS = 13;
  /** @brief log2 of a FlatMap's slots, four times the read slots a search holds */
  constexpr size_t FLATMAP_BITS = 9;

  /**
   * @brief An open-addressing set of integer keys a search reuses query after query: clear bumps a
   *        stamp instead of touching every slot, and the table doubles once half full
   * @tparam KEY integer key type
   */
  template<class KEY>
  class FlatSet {
   public:
    /** @brief An empty set of 2^FLAT_BITS slots */
    FlatSet();

    /** @brief Free the slots */
    ~FlatSet();

    FlatSet(const FlatSet &) = delete;
    FlatSet &operator=(const FlatSet &) = delete;

    /**
     * @brief Add key
     * @param key the key
     * @return whether key was absent
     */
    bool insert(KEY key);

    /**
     * @brief Whether key was added since the last clear
     * @param key the key
     * @return true when present
     */
    bool contains(KEY key) const;

    /** @brief Forget every key in O(1) */
    void clear();

    /**
     * @brief Keys added since the last clear
     * @return the count
     */
    size_t size() const;

   private:
    /** @brief Double the slots and re-add the live keys */
    void grow();

    KEY *keys;
    uint32_t *stamps;
    uint32_t stamp;
    size_t mask;
    size_t count;
  };

  /**
   * @brief An open-addressing map from integer keys to small values, cleared by a stamp, erasing by
   *        backward shift so no tombstone lengthens a probe; sized for the reads one search holds
   * @tparam KEY integer key type
   * @tparam VALUE trivially copyable value type
   */
  template<class KEY, class VALUE>
  class FlatMap {
   public:
    /** @brief An empty map of 2^FLATMAP_BITS slots */
    FlatMap();

    /** @brief Free the slots */
    ~FlatMap();

    FlatMap(const FlatMap &) = delete;
    FlatMap &operator=(const FlatMap &) = delete;

    /**
     * @brief The value under key
     * @param key the key
     * @return a pointer to the value, nullptr when absent
     */
    VALUE *find(KEY key);

    /**
     * @brief Put value under key, replacing any value there
     * @param key the key
     * @param value the value
     */
    void insert(KEY key, const VALUE &value);

    /**
     * @brief Drop key
     * @param key the key
     * @return whether key was present
     */
    bool erase(KEY key);

    /** @brief Forget every key in O(1) */
    void clear();

   private:
    /**
     * @brief The slot key hashes to
     * @param key the key
     * @return the home slot
     */
    size_t home(KEY key) const;

    KEY *keys;
    VALUE *values;
    uint32_t *stamps;
    uint32_t stamp;
  };

  /**
   * @brief A 64-bit mix of key, spreading consecutive ids over the slots
   * @tparam KEY integer key type
   * @param key the key
   * @return the mixed bits
   */
  template<class KEY>
  inline uint64_t flatmix(KEY key) {
    uint64_t h = (uint64_t) key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return h;
  }

  template<class KEY>
  FlatSet<KEY>::FlatSet()
      : keys(new KEY[(size_t) 1 << FLAT_BITS]), stamps(new uint32_t[(size_t) 1 << FLAT_BITS]()), stamp(1),
        mask(((size_t) 1 << FLAT_BITS) - 1), count(0) {
  }

  template<class KEY>
  FlatSet<KEY>::~FlatSet() {
    delete[] keys;
    delete[] stamps;
  }

  template<class KEY>
  bool FlatSet<KEY>::insert(KEY key) {
    // [1]
    if ((count + 1) * 2 > mask + 1) {
      grow();
    }
    // [2]
    for (size_t at = flatmix(key) & mask;; at = (at + 1) & mask) {
      if (stamps[at] != stamp) {
        stamps[at] = stamp;
        keys[at] = key;
        ++count;
        return true;
      }
      if (keys[at] == key) {
        return false;
      }
    }
  }

  template<class KEY>
  bool FlatSet<KEY>::contains(KEY key) const {
    // [1]
    for (size_t at = flatmix(key) & mask;; at = (at + 1) & mask) {
      if (stamps[at] != stamp) {
        return false;
      }
      if (keys[at] == key) {
        return true;
      }
    }
  }

  template<class KEY>
  void FlatSet<KEY>::clear() {
    // [1]
    count = 0;
    if (++stamp == 0) {
      std::memset(stamps, 0, (mask + 1) * sizeof(uint32_t));
      stamp = 1;
    }
  }

  template<class KEY>
  size_t FlatSet<KEY>::size() const {
    return count;
  }

  template<class KEY>
  void FlatSet<KEY>::grow() {
    // [1]
    const size_t old = mask + 1;
    KEY *okeys = keys;
    uint32_t *ostamps = stamps;
    const uint32_t ostamp = stamp;
    keys = new KEY[old * 2];
    stamps = new uint32_t[old * 2]();
    stamp = 1;
    mask = old * 2 - 1;
    count = 0;
    // [2]
    for (size_t i = 0; i < old; ++i) {
      if (ostamps[i] == ostamp) {
        insert(okeys[i]);
      }
    }
    delete[] okeys;
    delete[] ostamps;
  }

  template<class KEY, class VALUE>
  FlatMap<KEY, VALUE>::FlatMap()
      : keys(new KEY[(size_t) 1 << FLATMAP_BITS]), values(new VALUE[(size_t) 1 << FLATMAP_BITS]),
        stamps(new uint32_t[(size_t) 1 << FLATMAP_BITS]()), stamp(1) {
  }

  template<class KEY, class VALUE>
  FlatMap<KEY, VALUE>::~FlatMap() {
    delete[] keys;
    delete[] values;
    delete[] stamps;
  }

  template<class KEY, class VALUE>
  size_t FlatMap<KEY, VALUE>::home(KEY key) const {
    return flatmix(key) & (((size_t) 1 << FLATMAP_BITS) - 1);
  }

  template<class KEY, class VALUE>
  VALUE *FlatMap<KEY, VALUE>::find(KEY key) {
    // [1]
    const size_t mask = ((size_t) 1 << FLATMAP_BITS) - 1;
    for (size_t at = home(key);; at = (at + 1) & mask) {
      if (stamps[at] != stamp) {
        return nullptr;
      }
      if (keys[at] == key) {
        return &values[at];
      }
    }
  }

  template<class KEY, class VALUE>
  void FlatMap<KEY, VALUE>::insert(KEY key, const VALUE &value) {
    // [1]
    const size_t mask = ((size_t) 1 << FLATMAP_BITS) - 1;
    for (size_t at = home(key);; at = (at + 1) & mask) {
      if (stamps[at] != stamp) {
        stamps[at] = stamp;
        keys[at] = key;
        values[at] = value;
        return;
      }
      if (keys[at] == key) {
        values[at] = value;
        return;
      }
    }
  }

  template<class KEY, class VALUE>
  bool FlatMap<KEY, VALUE>::erase(KEY key) {
    // [1]
    const size_t mask = ((size_t) 1 << FLATMAP_BITS) - 1;
    size_t at = home(key);
    for (;; at = (at + 1) & mask) {
      if (stamps[at] != stamp) {
        return false;
      }
      if (keys[at] == key) {
        break;
      }
    }
    // [2] backward shift: pull later entries of the run into the hole when their home allows it
    for (size_t next = (at + 1) & mask;; next = (next + 1) & mask) {
      if (stamps[next] != stamp) {
        stamps[at] = 0;
        return true;
      }
      const size_t want = home(keys[next]);
      const bool movable = (at <= next) ? (want <= at || want > next) : (want <= at && want > next);
      if (movable) {
        keys[at] = keys[next];
        values[at] = values[next];
        at = next;
      }
    }
  }

  template<class KEY, class VALUE>
  void FlatMap<KEY, VALUE>::clear() {
    // [1]
    if (++stamp == 0) {
      std::memset(stamps, 0, ((size_t) 1 << FLATMAP_BITS) * sizeof(uint32_t));
      stamp = 1;
    }
  }
}  // namespace pipeann
