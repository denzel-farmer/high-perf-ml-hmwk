CC := gcc
CFLAGS := -Wall -Werror -g -O3
LDFLAGS :=


OBJECTS := dp1.o 

all: dp1


dp1: $(OBJECTS)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $^

clean:
	rm -f dp1 *.o

.PHONY: clean
