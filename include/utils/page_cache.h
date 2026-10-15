#ifndef PAGE_CACHE_H_
#define PAGE_CACHE_H_

#include <cstring>
#include <cstdint>
#include "utils.h"
#include "utils/libcuckoo/cuckoohash_map.hh"

namespace pipeann {
  // User-space page cache for update acceleration (in fact it's a buffer)
  // only used for write-write, ensure that disk has a consistent state
  // expect a lock-free read

  struct PageCacheItem {
    uint8_t *buf;
    uint64_t ref_cnt;

    // use lock!
    uint64_t ref() {
      return ++ref_cnt;
    }

    // use lock!
    uint64_t deref() {
      return --ref_cnt;
    }
  };

  union PageCacheKey {
    struct {
      uint32_t block_no;
      int fd;
    };
    uint64_t raw;

    PageCacheKey(int fd, uint32_t block_no) : block_no(block_no), fd(fd) {
    }

    static PageCacheKey from_raw(uint64_t raw) {
      return PageCacheKey(raw);
    }

   private:
    PageCacheKey(uint64_t raw) : raw(raw) {
    }
  };

  struct PageCache {
    // max file size: SECTOR_LEN * 4GB = 16TB.
    bool get(PageCacheKey key, uint8_t *value, bool ref = false) {
      bool ret = cache.update_fn(key.raw, [&](PageCacheItem &v) {
        memcpy(value, v.buf, SECTOR_LEN);
        if (ref) {
          v.ref();
        }
      });
      return ret;
    }

    bool put(PageCacheKey key, uint8_t *value, bool ref = false) {
      return cache.upsert(key.raw, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = PageCacheItem{.buf = new uint8_t[SECTOR_LEN], .ref_cnt = 0};
        }
        if (ref) {
          v.ref();
        }
        memcpy(v.buf, value, SECTOR_LEN);
      });
    }

    // Cache a page just read from disk. An entry already there is newer than the disk (a write puts
    // it before its disk write lands), so it is kept and copied back into value instead.
    bool fill(PageCacheKey key, uint8_t *value, bool ref = false) {
      return cache.upsert(key.raw, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          v = PageCacheItem{.buf = new uint8_t[SECTOR_LEN], .ref_cnt = 0};
          memcpy(v.buf, value, SECTOR_LEN);
        } else {
          memcpy(value, v.buf, SECTOR_LEN);
        }
        if (ref) {
          v.ref();
        }
      });
    }

    bool deref(PageCacheKey key) {
      bool ret = cache.uprase_fn(key.raw, [&](PageCacheItem &v, libcuckoo::UpsertContext ctx) {
        if (ctx == libcuckoo::UpsertContext::NEWLY_INSERTED) {
          LOG(ERROR) << "PageCache: deref a non-exist fd: " << key.fd << " block_no: " << key.block_no;
          return true;
          __builtin_trap();
        }
        uint64_t refs = v.deref();
        if (refs == 0) {
          delete[] v.buf;
        }
        return refs == 0;
      });
      return ret;
    }

    void clear() {
      cache.clear();
    }
    libcuckoo::cuckoohash_map<uint64_t, PageCacheItem> cache;
  };

  inline PageCache cache;
}  // namespace pipeann

#endif  // PAGE_CACHE_H_