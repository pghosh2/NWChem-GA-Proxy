ifeq ($(USER),jhammond)
    MPILOC		= /opt/local/bin
    CFLAGS 		= -O3 -std=gnu99 -Wall -pedantic -DUSE_GSL_CBLAS -DDEBUG=1000
    LDFLAGS		= -lgslcblas
else
    MPILOC		= /opt/bin
    CFLAGS 		= -O3 -std=gnu99 -Wall -pedantic -DDEBUG=1 
    LDFLAGS		= -lcblas
endif

MPICC   	= $(MPILOC)/mpicc
SRCS 		= matmul.c
NAME 		= Xmatmul_X
INCLUDE 	= -I.
NPROCS 		= 4
#data--->
M		= 200
N		= 200
K		= 200
bX		= 100
bY		= 100

.PHONY: all clean run ga

all: $(NAME) ga

ga:
	$(MPICC) $(INCLUDE) -D_GNU_SOURCE $(CFLAGS) -c fakega.c

$(NAME) : $(SRCS) ga
	$(MPICC) $(INCLUDE) -D_GNU_SOURCE $(CFLAGS) -o $(NAME) $(SRCS) fakega.o $(LDFLAGS) -lm

run:
	$(MPILOC)/mpiexec -np $(NPROCS) ./$(NAME) $(M) $(N) $(K) $(bX) $(bY)
	
clean:
	$(RM) *.o $(NAME)
