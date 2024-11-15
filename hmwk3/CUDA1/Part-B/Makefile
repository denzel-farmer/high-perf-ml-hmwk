SDK_INSTALL_PATH := /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
OPTIONS   :=  -O3 
EXECS :=  q1 q2 q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

q1: q1.cpp utils.cpp
	g++ q1.cpp utils.cpp -o q1 $(OPTIONS)

q2 : q2.cu utils.cpp
	${NVCC} q2.cu utils.cpp -o q2 $(LIB) $(OPTIONS)

q3 : q3.cu utils.cpp
	${NVCC} q3.cu utils.cpp -o q3 $(LIB) $(OPTIONS)
