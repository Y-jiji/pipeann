#pragma once

#include <pthread.h>

namespace pipeann {
  /**
   * @brief A shared mutex whose waiting writer holds new readers off, unlike glibc's default
   * reader-preferring rwlock behind std::shared_mutex, so a merge's brief exclusive section gets in
   * while searches keep arriving; shared holds must not nest on one thread
   */
  class WriterFirstMutex {
   public:
    /** @brief A writer-preferring, non-recursive pthread rwlock */
    WriterFirstMutex();

    /** @brief Destroy the rwlock */
    ~WriterFirstMutex();

    WriterFirstMutex(const WriterFirstMutex &) = delete;
    WriterFirstMutex &operator=(const WriterFirstMutex &) = delete;

    /** @brief Hold it exclusively, new readers waiting behind */
    void lock();

    /**
     * @brief Hold it exclusively if free
     * @return whether it was taken
     */
    bool try_lock();

    /** @brief Release an exclusive hold */
    void unlock();

    /** @brief Hold it shared, waiting behind any waiting writer */
    void lock_shared();

    /**
     * @brief Hold it shared if no writer holds or waits
     * @return whether it was taken
     */
    bool try_lock_shared();

    /** @brief Release a shared hold */
    void unlock_shared();

   private:
    pthread_rwlock_t handle;
  };

  inline WriterFirstMutex::WriterFirstMutex() {
    // [1]
    pthread_rwlockattr_t attr;
    pthread_rwlockattr_init(&attr);
    pthread_rwlockattr_setkind_np(&attr, PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
    pthread_rwlock_init(&handle, &attr);
    pthread_rwlockattr_destroy(&attr);
  }

  inline WriterFirstMutex::~WriterFirstMutex() {
    pthread_rwlock_destroy(&handle);
  }

  inline void WriterFirstMutex::lock() {
    pthread_rwlock_wrlock(&handle);
  }

  inline bool WriterFirstMutex::try_lock() {
    return pthread_rwlock_trywrlock(&handle) == 0;
  }

  inline void WriterFirstMutex::unlock() {
    pthread_rwlock_unlock(&handle);
  }

  inline void WriterFirstMutex::lock_shared() {
    pthread_rwlock_rdlock(&handle);
  }

  inline bool WriterFirstMutex::try_lock_shared() {
    return pthread_rwlock_tryrdlock(&handle) == 0;
  }

  inline void WriterFirstMutex::unlock_shared() {
    pthread_rwlock_unlock(&handle);
  }
}  // namespace pipeann
