#!/bin/bash
# Full enwik8 benchmark: all CPGC levels (compress + verified decompress),
# then local gzip/bzip2/xz references. Emits CSV.
set -u
cd "$(dirname "$0")"
BIN=../target/release/cpgc
CSV=results.csv
echo "mode,comp_bytes,comp_seconds,decomp_seconds,verified" > $CSV

now() { date +%s.%N; }

for L in 1 2 3 4 5 6 7 8 9; do
  t0=$(now)
  $BIN compress enwik8 enwik8.l$L.cpgc -l $L >/dev/null 2>&1
  t1=$(now)
  if $BIN verify enwik8.l$L.cpgc >/dev/null 2>&1; then ok=1; else ok=0; fi
  t2=$(now)
  size=$(stat -c%s enwik8.l$L.cpgc)
  echo "cpgc-$L,$size,$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV
  echo "done level $L: $size bytes" >&2
done

# Classical tools at max settings (compress + decompress timing).
t0=$(now); gzip -9 -c enwik8 > enwik8.gz; t1=$(now)
gzip -dc enwik8.gz | cmp -s - enwik8 && ok=1 || ok=0; t2=$(now)
echo "gzip-9,$(stat -c%s enwik8.gz),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV

t0=$(now); bzip2 -9 -c enwik8 > enwik8.bz2; t1=$(now)
bzip2 -dc enwik8.bz2 | cmp -s - enwik8 && ok=1 || ok=0; t2=$(now)
echo "bzip2-9,$(stat -c%s enwik8.bz2),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV

t0=$(now); xz -9e -T0 -c enwik8 > enwik8.xz; t1=$(now)
xz -dc enwik8.xz | cmp -s - enwik8 && ok=1 || ok=0; t2=$(now)
echo "xz-9e,$(stat -c%s enwik8.xz),$(awk "BEGIN{printf \"%.1f\", $t1-$t0}"),$(awk "BEGIN{printf \"%.1f\", $t2-$t1}"),$ok" >> $CSV

echo "ALL DONE" >&2
cat $CSV
