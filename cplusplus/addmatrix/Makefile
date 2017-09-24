CC = clang -c
#CC = gcc -c
LD = clang
#LD = gcc

NAME = addmatrix

INC_DIRS = -I/usr/local/include/tensorflow

CPPFLAGS = -Wall -std=c++11
LDFLAGS = -lstdc++ -ltensorflow_cc

all: release

release: $(NAME)

OBJS = main.o

%.o : %.cpp
	$(CC) -o $@ $(INC_DIRS) $(CPPFLAGS) $<

$(NAME) : $(OBJS)
	$(LD) -o $@ $(LDFLAGS) $^

clean:
	rm -rf *~ *.o $(NAME)
