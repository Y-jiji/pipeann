#pragma once
// Instrumentation-only trace sink for pipe_search_common.h. Compiled in
// only under -DPIPE_TRACE; every call site is guarded so a normal build
// carries neither the branch nor the symbol.
//
// One line per event, appended to $PIPE_TRACE_FILE. The grammar is
// documented at the emit sites in pipe_search_common.h.

#ifdef PIPE_TRACE
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace pipeann {
  /// Trace sink, opened once from $PIPE_TRACE_FILE. Null when the
  /// variable is unset, which turns every PTRACE into a no-op.
  inline FILE *ptrace_sink() {
    static FILE *sink = [] {
      const char *path = std::getenv("PIPE_TRACE_FILE");
      return path == nullptr ? (FILE *) nullptr : std::fopen(path, "w");
    }();
    return sink;
  }

  /// Append one trace line.
  inline void ptrace_emit(const char *fmt, ...) {
    FILE *sink = ptrace_sink();
    if (sink == nullptr) {
      return;
    }
    va_list args;
    va_start(args, fmt);
    std::vfprintf(sink, fmt, args);
    va_end(args);
    std::fputc('\n', sink);
  }
}  // namespace pipeann

#define PTRACE(...) ::pipeann::ptrace_emit(__VA_ARGS__)
#else
#define PTRACE(...) ((void) 0)
#endif
