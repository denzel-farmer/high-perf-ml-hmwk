CC := gcc
CFLAGS := -Wall -g -O3
LDFLAGS :=

PROGRAMS := dp1 dp2 dp3

all: $(PROGRAMS)

dp3: dp3.c
	/opt/intel/oneapi/mkl/latest/bin/mkl_link_tool $(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< 

%: %.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(PROGRAMS) *.o

.PHONY: all clean

