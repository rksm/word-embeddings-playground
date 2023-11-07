#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# cargo run --features cuda --release -- -f data/w2v_skipgram.nn --method skipgram

cargo run --features cuda --release -- -f data/w2v_skipgram.nn --method skip-gram


# i=0
# while true; do
#     # cargo run --features cuda --release --example dot-product-candle
#     cargo run --features cuda --release
#     sleep 1
#     i=$((i+1))
#     echo $i
#     if [ $i -eq 3 ]; then
#         break
#     fi
# done
