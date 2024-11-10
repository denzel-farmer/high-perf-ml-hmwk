#!/bin/sh
set +x

make clean
make 

mv results.csv results.csv.old

for param in 1 5 10 50 100; do
    ./q1 $param
    ./q2 $param
    ./q3 $param
done