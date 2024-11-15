# Building
All programs can be compiled with `make` and cleaned with `make clean`

# P1: Vector Addition
Run experiment with `./P1-tests.sh`
## vecadd
Provided vector addition framework

## vecaddKernel00
Provided naive vector addition kernel implementation

## vecaddKernel01
Provided coalesced vector addition implementation

## P1-tests.sh 
- Cleans and rebuilds all files 
- Runs vecadd00 with sizes 16, 32, 64, giving each run a warmup (overall sizes 256,512,1024)
- Runs vecadd01 in the same manner
- Both runs print relevant times and throughput, showing the coalesced version is much faster

# P2 Matrix Multiplication
Run experiment with `./P2-tests.sh`

## matmult
Provided matrix multiplication framework

## matmultKernel00
Provided tiled matrix multiplication kernel

## P2-tests.sh
- Cleans and rebuilds all files 
- Runs matmult00 with sizes 8, 16, 32, giving each run a warmup (overalsizes 256,512,1024)
- Runs matmult01 in the same manner
- Both runs print relevant times and throughput, showing the coalesced version is much faster
