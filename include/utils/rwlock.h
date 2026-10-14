#pragma once

#include <pthread.h>

namespace pipeann {
  // A shared mutex whose waiting writer holds new readers off, unlike glibc's default
  // reader-preferring rwlock behind std::shared_mutex, so a merge's brief exclusive section
  // gets in while searches keep arriving. Shared holds must not nest on one thread.
  class WriterFirstMutex {
   public:
    WriterFirstMutex() {
      pthread_rwlockattr_t attr;
      pthread_rwlockattr_init(&attr);
      pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
      pthread_rwlock_init(&handle, &attr);
      pthread_rwlockattr_destroy(&attr);
    }

    ~WriterFirstMutex() {
      pthread_rwlock_destroy(&handle);
    }

    WriterFirstMutex(const WriterFirstMutex &) = delete;
    WriterFirstMutex &operator=(const WriterFirstMutex &) = delete;

    void lock() {
      pthread_rwlock_wrlock(&handle);
    }

    bool try_lock() {
      return pthread_rwlock_trywrlock(&handle) == 0;
    }

    void unlock() {
      pthread_rwlock_unlock(&handle);
    }

    void lock_shared() {
      pthread_rwlock_rdlock(&handle);
    }

    bool try_lock_shared() {
      return pthread_rwlock_tryrdlock(&handle) == 0;
    }

    void unlock_shared() {
      pthread_rwlock_unlock(&handle);
    }

   private:
    pthread_rwlock_t handle;
  };
}  // namespace pipeann
