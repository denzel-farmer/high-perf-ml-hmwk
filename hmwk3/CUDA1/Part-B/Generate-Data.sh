#!/bin/sh
set +x

echo "Rebuilding..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1 || { echo "Build failed"; exit 1; }

mv results.csv results.csv.old

for param in 1 5 10 50 100; do
echo "Running with param = $param"
echo "\nQ1 (CPU):"
    ./q1 $param
echo "\nQ2 (GPU):"
    ./q2 $param
echo "\nQ3 (GPU):"
    ./q3 $param
done