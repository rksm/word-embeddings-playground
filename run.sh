#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

cargo run --release --example dot-product-candle
