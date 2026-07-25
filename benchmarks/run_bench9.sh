#!/bin/bash
# enwik9 benchmark: a representative subset of levels, compress + verified
# decompress. Emits CSV.
#
#   DATA=/path/to/enwik9 LEVELS="1 3 5 9" ./run_bench9.sh
#
# Memory notes for a 1 GB input on a 15 GB machine:
#   * level 9 compresses the whole file as one segment — about 11 GB, single
#     threaded. Run it with nothing else on the box.
#   * level 8 keeps 64 MiB segments but its models are ~5 GB per worker, so
#     cap the pool (RAYON_NUM_THREADS=2) or it will exhaust memory.
#   * levels 1-5 are small enough to use every core.
set -u
cd "$(dirname "$0")"
BIN=${BIN:-../target/release/cpgc}
DATA=${DATA:-enwik9}
WORK=${WORK:-.}
CSV=${CSV:-enwik9_results.csv}
LEVELS=${LEVELS:-"1 3 5 9"}
echo "mode,comp_bytes,comp_seconds,decomp_seconds,verified" > $CSV

now() { date +%s.%N; }

for L in $LEVELS; do
  out=$WORK/enwik9.l$L.cpgc
  t0=$(now)
  $BIN compress "$DATA" "$out" -l $L >/dev/null 2>&1
  t1=$(now)
  if $BIN verify "$out" >/dev/null 2>&1; then ok=1; else ok=0; fi
  t2=$(now)
  size=$(stat -c%s "$out")
  echo "cpgc-$L,$size,$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV
  echo "done level $L: $size bytes (verified=$ok)" >&2
  rm -f "$out"
done

echo "ALL DONE" >&2
cat $CSV
