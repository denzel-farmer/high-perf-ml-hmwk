#!/bin/sh
set +x

make clean
make 

echo "(Q3) Naive implementation:"

./matmult00 16
./matmult00 32
./matmult00 64

echo ""
echo "(Q4) Coalesced Implementation:"

./matmult01 8
./matmult01 16
./matmult01 32