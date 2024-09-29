CC := gcc
CFLAGS := -Wall -g -O3
LDFLAGS :=

PROGRAMS := dp1 dp2 

all: $(PROGRAMS)

%: %.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(PROGRAMS) *.o

.PHONY: all clean

