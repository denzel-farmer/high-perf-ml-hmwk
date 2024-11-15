#!/bin/sh
set +x

echo "Rebuilding..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1 || { echo "Build failed"; exit 1; }

echo "(Q1) Naive implementation:"
./vecadd00 500 > /dev/null 2>&1
./vecadd00 500
./vecadd00 1000 > /dev/null 2>&1
./vecadd00 1000
./vecadd00 2000 > /dev/null 2>&1
./vecadd00 2000

echo ""
echo "(Q2) Coalesced Implementation:"
./vecadd01 500 > /dev/null 2>&1
./vecadd01 500
./vecadd01 1000 > /dev/null 2>&1
./vecadd01 1000
./vecadd01 2000 > /dev/null 2>&1
./vecadd01 2000



