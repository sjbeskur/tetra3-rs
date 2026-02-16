#!/bin/bash
set -e

cargo build --release --example tetrars

valgrind \
  --leak-check=full \
  --track-origins=yes \
  --error-exitcode=1 \
  ./target/release/examples/tetrars data/tetra_db/ data/2019-07-29T204726_Alt60_Azi-135_Try1.tiff --cal data/test_cal.toml