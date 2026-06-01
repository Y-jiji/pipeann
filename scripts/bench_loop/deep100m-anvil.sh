#!/usr/bin/env bash
# bench_loop on DEEP 100M, Anvil scratch paths.
# Mirrors fnct-bench/config/deep100m-odindb-anvil.toml.
TAG="deep100m-anvil"
DTYPE=float
DIM=96
LEN=100000000
BATCH=20000000
QLEN=10000
DATA="${DATA:-/anvil/scratch/x-tyang19/data/deep.100M.96.f32.bin}"
QUERY="${QUERY:-/anvil/scratch/x-tyang19/data/deep.10K.96.f32.query}"
INDEX_PREFIX="${INDEX_PREFIX:-/anvil/scratch/x-tyang19/index/pipeann_bench_loop_deep100m}"
OUT_PREFIX="${OUT_PREFIX:-/anvil/scratch/x-tyang19/result/result_deep100m_pipeann/}"
L_INSERT=128
R=32
ALPHA=1.2
INSERT_THREADS=120
SEARCH_THREADS=120
SEARCH_PAIRS=( 1 64  1 128  10 128  10 256 )
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
