#pragma once
// Instrumentation-only trace sinks for pipe_search_common.h. Compiled in only
// under -DPIPE_TRACE; every call site is guarded so a normal build carries
// neither the branch nor the symbol.
//
// Two sinks, because the records fall into two kinds and a replay must not see
// the second while producing its own:
//
//   PTRACE_IN  ($PIPE_TRACE_IN)  what the search was given -- query
//                                parameters, the seed pool, the graph it was
//                                shown.
//   PTRACE_REF ($PIPE_TRACE_REF) what the search decided -- reads issued,
//                                completions reaped per poll, beam width,
//                                termination.
//
// Splitting here rather than by filtering the log afterwards keeps the
// distinction where it is actually known.

#ifdef PIPE_TRACE
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace pipeann {
  inline FILE *ptrace_open(const char *var) {
    const char *path = std::getenv(var);
    return path == nullptr ? (FILE *) nullptr : std::fopen(path, "w");
  }

  inline FILE *ptrace_in() {
    static FILE *sink = ptrace_open("PIPE_TRACE_IN");
    return sink;
  }

  inline FILE *ptrace_ref() {
    static FILE *sink = ptrace_open("PIPE_TRACE_REF");
    return sink;
  }

  /// Append one record to `sink`, if it is open.
  inline void ptrace_emit(FILE *sink, const char *fmt, ...) {
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

#define PTRACE_IN(...) ::pipeann::ptrace_emit(::pipeann::ptrace_in(), __VA_ARGS__)
#define PTRACE_REF(...) ::pipeann::ptrace_emit(::pipeann::ptrace_ref(), __VA_ARGS__)
#else
#define PTRACE_IN(...) ((void) 0)
#define PTRACE_REF(...) ((void) 0)
#endif
