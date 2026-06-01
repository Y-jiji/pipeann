#!/usr/bin/env bash
# bench_loop on SIFT 10M, local T9 paths (laptop smoke test).
# Mirrors fnct-bench/config/sift10m.toml.
TAG="sift10m-local"
DTYPE=uint8
DIM=128
LEN=10000000
BATCH=100000
QLEN=10000
DATA="${DATA:-/media/tj-yang/T9/data/sift.1B.128.u8.bin}"
QUERY="${QUERY:-/media/tj-yang/T9/data/sift.10K.128.u8.query}"
INDEX_PREFIX="${INDEX_PREFIX:-/media/tj-yang/T9/index/pipeann_bench_loop_sift10m}"
OUT_PREFIX="${OUT_PREFIX:-/media/tj-yang/T9/result/result_sift10m_pipeann/}"
L_INSERT=128
R=32
ALPHA=1.2
INSERT_THREADS=12
SEARCH_THREADS=12
SEARCH_PAIRS=( 1 64  1 128  10 128  10 256 )
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
