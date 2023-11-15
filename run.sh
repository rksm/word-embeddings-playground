#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# cargo run --features cuda --release -- -f data/doppelgaenger/cbow2.nn --method cbow

cargo build --features cuda --release
# ./target/perf record -o ./target/perf.data --call-graph=lbr ./target/release/word-embeddings-playground -f data/doppelgaenger/cbow2.nn --method cbow
./target/perf record --call-graph=dwarf -o ./target/perf.data ./target/release/word-embeddings-playground -f data/doppelgaenger/cbow2.nn --method cbow

# cargo run --features cuda --release -- -f data/doppelgaenger/w2v_skipgram4.nn --method skip-gram
# cargo flamegraph --features cuda -- --method skip-gram


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
