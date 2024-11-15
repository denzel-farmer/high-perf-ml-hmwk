#!/bin/sh
set +x

echo "Rebuilding..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1 || { echo "Build failed"; exit 1; }

echo "(Q3) Naive implementation:"
./matmult00 16 > /dev/null 2>&1
./matmult00 16
./matmult00 32 > /dev/null 2>&1
./matmult00 32
./matmult00 64 > /dev/null 2>&1
./matmult00 64

echo ""
echo "(Q4) Coalesced Implementation:"

./matmult01 8 > /dev/null 2>&1
./matmult01 8
./matmult01 16 > /dev/null 2>&1
./matmult01 16
./matmult01 32 > /dev/null 2>&1
./matmult01 32