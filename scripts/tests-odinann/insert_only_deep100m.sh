#!/usr/bin/env bash
# Insert-only OdinANN benchmark on DEEP 100M, mirroring fnct-bench's
# `deep100m-odindb-anvil.toml` experiment: start with an empty index,
# insert 20M vectors per batch (5 batches total) and run search after
# each batch with the same (k, L) configurations.
#
# Result files land next to fnct-bench's outputs (one .txt per search
# config, fnct-bench's text format) and can be compared with
# `fnct-bench/script/recall.py` against the brute-force ground truth.
#
# Build prerequisites (see refs/pipeann/README.md):
#   cd refs/pipeann && bash build.sh
#
# Run from refs/pipeann/ (or any cwd; paths are absolute).

set -euo pipefail

# ---------------- paths (override via env) -------------------------
DATA="${DATA:-/anvil/scratch/x-tyang19/data/deep.100M.96.f32.bin}"
QUERY="${QUERY:-/anvil/scratch/x-tyang19/data/deep.10K.96.f32.query}"
INDEX_PREFIX="${INDEX_PREFIX:-/anvil/scratch/x-tyang19/index/pipeann_odinann_deep100m}"
OUT_PREFIX="${OUT_PREFIX:-/anvil/scratch/x-tyang19/result/result_deep100m_pipeann_odinann/}"

# Resolve repo root so we can find the built binary regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPEANN_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BIN="${PIPEANN_ROOT}/build/tests/test_insert_only"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}" >&2
  echo "Build it first: (cd ${PIPEANN_ROOT} && bash build.sh)" >&2
  exit 1
fi

# ---------------- dataset / experiment params ----------------------
DTYPE=float
DIM=96
TOTAL_LEN=100000000   # 100M (matches [dataset].len)
BATCH=20000000        # 20M (matches [dataset].batch)
QLEN=10000            # 10K (matches [dataset].qlen)

# ---------------- index params (mirror [odindb]) -------------------
L_INSERT=128          # insert visitor: beamest(k=128,beam=128,starts=1) → L=128
MAX_DEGREE=32         # [odindb].max_degree
ALPHA=1.2             # [odindb].alpha

# ---------------- threading (mirror [runtime]) ---------------------
INSERT_THREADS=120
SEARCH_THREADS=120

# ---------------- search configs (mirror [visitor].search) ---------
# Each pair: <k> <L>
SEARCH_PAIRS=(
  1 64
  1 128
  10 128
  10 256
)

# out_prefix is used literally as a file/path prefix (matches
# fnct-bench's file-prefix semantics). Create the parent of the
# resolved stats file — works for both `.../dir/` (parent is `.../dir`)
# and `.../dir/pipeann_` (parent is `.../dir`).
mkdir -p "$(dirname "${OUT_PREFIX}stats.tsv")"
mkdir -p "$(dirname "${INDEX_PREFIX}_disk.index")"

# Clean any prior shadow/merge artifacts from a previous run on the
# same prefix (mirrors what fig6/fig7 scripts do for test_insert_search).
rm -f "${INDEX_PREFIX}"_shadow* "${INDEX_PREFIX}"_merge* \
      "${INDEX_PREFIX}"_mem* "${INDEX_PREFIX}"_disk* \
      "${INDEX_PREFIX}"_v2* "${INDEX_PREFIX}"temp0* 2>/dev/null || true

echo "=== insert-only OdinANN on DEEP 100M ==="
echo "  data       : ${DATA}"
echo "  query      : ${QUERY}"
echo "  index pfx  : ${INDEX_PREFIX}"
echo "  out pfx    : ${OUT_PREFIX}"
echo "  total/batch: ${TOTAL_LEN} / ${BATCH}"
echo "  R/L/alpha  : ${MAX_DEGREE} / ${L_INSERT} / ${ALPHA}"
echo "  threads    : insert=${INSERT_THREADS} search=${SEARCH_THREADS}"
echo "  search     : ${SEARCH_PAIRS[*]}"
echo

exec "${BIN}" \
  "${DTYPE}" "${DATA}" "${DIM}" "${TOTAL_LEN}" "${BATCH}" \
  "${QUERY}" "${QLEN}" \
  "${INDEX_PREFIX}" "${OUT_PREFIX}" \
  "${L_INSERT}" "${MAX_DEGREE}" "${ALPHA}" \
  "${INSERT_THREADS}" "${SEARCH_THREADS}" \
  "${SEARCH_PAIRS[@]}"
