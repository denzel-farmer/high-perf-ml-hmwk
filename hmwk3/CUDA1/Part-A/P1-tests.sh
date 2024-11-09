#!/bin/sh
set +x

make clean
make 

echo "(Q1) Naive implementation:"

./vecadd00 500
./vecadd00 1000
./vecadd00 2000

echo ""
echo "(Q2) Coalesced Implementation:"

./vecadd01 500
./vecadd01 1000
./vecadd01 2000



