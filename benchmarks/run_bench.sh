#!/bin/bash
# Full enwik8 benchmark: all CPGC levels (compress + verified decompress),
# then local gzip/bzip2/xz references. Emits CSV.
#
#   DATA=/path/to/enwik8 ./run_bench.sh
#
# Level 9 compresses the whole file as one segment and needs roughly 9 GB;
# it is single threaded by construction, so it also takes the longest.
set -u
cd "$(dirname "$0")"
BIN=${BIN:-../target/release/cpgc}
DATA=${DATA:-enwik8}
WORK=${WORK:-.}
CSV=${CSV:-results.csv}
echo "mode,comp_bytes,comp_seconds,decomp_seconds,verified" > $CSV

now() { date +%s.%N; }

for L in 1 2 3 4 5 6 7 8 9; do
  out=$WORK/enwik8.l$L.cpgc
  t0=$(now)
  $BIN compress "$DATA" "$out" -l $L >/dev/null 2>&1
  t1=$(now)
  if $BIN verify "$out" >/dev/null 2>&1; then ok=1; else ok=0; fi
  t2=$(now)
  size=$(stat -c%s "$out")
  echo "cpgc-$L,$size,$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV
  echo "done level $L: $size bytes" >&2
  rm -f "$out"
done

# Classical tools at max settings (compress + decompress timing).
t0=$(now); gzip -9 -c "$DATA" > $WORK/enwik8.gz; t1=$(now)
gzip -dc $WORK/enwik8.gz | cmp -s - "$DATA" && ok=1 || ok=0; t2=$(now)
echo "gzip-9,$(stat -c%s $WORK/enwik8.gz),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV

t0=$(now); bzip2 -9 -c "$DATA" > $WORK/enwik8.bz2; t1=$(now)
bzip2 -dc $WORK/enwik8.bz2 | cmp -s - "$DATA" && ok=1 || ok=0; t2=$(now)
echo "bzip2-9,$(stat -c%s $WORK/enwik8.bz2),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV

t0=$(now); xz -9e -T0 -c "$DATA" > $WORK/enwik8.xz; t1=$(now)
xz -dc $WORK/enwik8.xz | cmp -s - "$DATA" && ok=1 || ok=0; t2=$(now)
echo "xz-9e,$(stat -c%s $WORK/enwik8.xz),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV
rm -f $WORK/enwik8.gz $WORK/enwik8.bz2 $WORK/enwik8.xz

echo "ALL DONE" >&2
cat $CSV
