SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS   :=  -O3
EXECS :=  c1 c2 c3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

c1: c1.cu ImageUtils.o
	${NVCC} $< ImageUtils.o -o $@ $(OPTIONS)

c2: c2.cu ImageUtils.o
	${NVCC} $< ImageUtils.o -o $@ $(OPTIONS)

c3: c3.cu
	${NVCC} $< -o $@ $(OPTIONS) -lcudnn

ImageUtils.o: ImageUtils.cu ImageUtils.h
	${NVCC} ImageUtils.cu -c -o ImageUtils.o $(OPTIONS)

