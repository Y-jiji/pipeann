#!/usr/bin/env bash
# bench_loop on SIFT 100M, Anvil scratch paths.
# Mirrors fnct-bench/config/sift100m-dpodb-anvil.toml.
TAG="sift100m-anvil"
DTYPE=uint8
DIM=128
LEN=100000000
BATCH=20000000
QLEN=10000
DATA="${DATA:-/anvil/scratch/x-tyang19/data/sift.100M.128.u8.bin}"
QUERY="${QUERY:-/anvil/scratch/x-tyang19/data/sift.10K.128.u8.query}"
INDEX_PREFIX="${INDEX_PREFIX:-/anvil/scratch/x-tyang19/index/pipeann_bench_loop_sift100m}"
OUT_PREFIX="${OUT_PREFIX:-/anvil/scratch/x-tyang19/result/result_sift100m_pipeann/}"
L_INSERT=128
R=32
ALPHA=1.2
INSERT_THREADS=120
SEARCH_THREADS=120
SEARCH_PAIRS=( 1 64  1 128  10 128  10 256 )
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
