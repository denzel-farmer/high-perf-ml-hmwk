#!/bin/sh
set +x

exec > >(tee -i program-output.txt) 2>&1

echo "Rebuilding..."
make clean > /dev/null 2>&1
make > /dev/null 2>&1 || { echo "Build failed"; exit 1; }


echo "(C1) Running simple convolution:"
for i in 1 2 3 4; do
    ./c1 0 > /dev/null 2>&1
done
./c1

echo ""
echo "(C2) Running shared memory convolution:"
for i in 1 2 3 4; do
    ./c2 0 > /dev/null 2>&1
done
./c2

echo ""
echo "(C3) Running cuDNN convolution:"
for i in 1 2 3 4; do
    ./c3 0 > /dev/null 2>&1
done
./c3