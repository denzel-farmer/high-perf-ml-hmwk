#!/bin/sh
set +x

make clean
make 

./vecadd00 500
./vecadd00 1000
./vecadd00 2000




