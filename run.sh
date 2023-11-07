#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

i=0

while true; do
    cargo run --release --example dot-product-candle
    sleep 1
    i=$((i+1))
    echo $i
    if [ $i -eq 3 ]; then
        break
    fi
done
