#!/usr/bin/env bash
# Shared launcher for bench_loop scripts. Sourced by per-dataset wrappers;
# resolves the built binary, makes parent dirs, cleans stale index files,
# prints config, and execs the binary with all positional args baked in.
#
# Each per-dataset script sets these env vars (with sensible defaults that
# can be overridden by `VAR=... ./script.sh`) before sourcing this file:
#   DTYPE, DATA, QUERY, DIM, LEN, BATCH, QLEN
#   INDEX_PREFIX, OUT_PREFIX
#   L_INSERT, R, ALPHA, INSERT_THREADS, SEARCH_THREADS
#   SEARCH_PAIRS=( k1 L1 k2 L2 ... )

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
PIPEANN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN="${PIPEANN_ROOT}/build/tests/bench_loop"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}" >&2
  echo "Build it first: (cd ${PIPEANN_ROOT} && bash build.sh)" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PREFIX}records.csv")"
mkdir -p "$(dirname "${INDEX_PREFIX}_disk.index")"

# Mirror tests-odinann cleanup of stale shadow/merge/mem artifacts.
rm -f "${INDEX_PREFIX}"_shadow* "${INDEX_PREFIX}"_merge* \
      "${INDEX_PREFIX}"_mem* "${INDEX_PREFIX}"_disk* \
      "${INDEX_PREFIX}"_v2* "${INDEX_PREFIX}"temp0* 2>/dev/null || true

echo "=== bench_loop: ${TAG:-unnamed} ==="
echo "  data       : ${DATA}"
echo "  query      : ${QUERY}"
echo "  index pfx  : ${INDEX_PREFIX}"
echo "  out pfx    : ${OUT_PREFIX}"
echo "  dtype/dim  : ${DTYPE} / ${DIM}"
echo "  len/batch  : ${LEN} / ${BATCH}"
echo "  qlen       : ${QLEN}"
echo "  R/L/alpha  : ${R} / ${L_INSERT} / ${ALPHA}"
echo "  threads    : insert=${INSERT_THREADS} search=${SEARCH_THREADS}"
echo "  search     : ${SEARCH_PAIRS[*]}"
echo

exec "${BIN}" \
  "${DTYPE}" "${DATA}" "${DIM}" "${LEN}" "${BATCH}" \
  "${QUERY}" "${QLEN}" \
  "${INDEX_PREFIX}" "${OUT_PREFIX}" \
  "${L_INSERT}" "${R}" "${ALPHA}" \
  "${INSERT_THREADS}" "${SEARCH_THREADS}" \
  "${SEARCH_PAIRS[@]}"
