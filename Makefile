# As of ga-5-1, the variables needed to build GA are determined from the
# ga-config script which is installed with GA.

ifndef GA_CONFIG
error:
	echo "you must set GA_CONFIG e.g. GA_CONFIG=/path/to/ga-config"
	exit 1
endif

CC       = $(shell $(GA_CONFIG) --cc)
F77      = $(shell $(GA_CONFIG) --f77)
CFLAGS   = $(shell $(GA_CONFIG) --cflags)
FFLAGS   = $(shell $(GA_CONFIG) --fflags)
CPPFLAGS = $(shell $(GA_CONFIG) --cppflags)
LDFLAGS  = $(shell $(GA_CONFIG) --ldflags)
LIBS     = $(shell $(GA_CONFIG) --libs)
FLIBS    = $(shell $(GA_CONFIG) --flibs)

# =========================================================================== 

FFLAGS += -O -g
CFLAGS += -O2 -g -std=gnu99 -I/soft/libraries/unsupported/atlas-3.10.1/include -DDEBUG=0 -DVERIFY=2
CPPFLAGS += -DUSE_MPI -DMPI

LINK = $(F77) -L/soft/libraries/unsupported/atlas-3.10.1/lib #-lm
LOADER_OPTS = -g
LIBS += -lcblas -latlas
#LIBS += -lgsl -lgslcblas

PROGRAMS =
PROGRAMS += spiral-mm-v1.c
PROGRAMS += spiral-mm-v1.x
PROGRAMS += test-mm-v1.c
PROGRAMS += test-mm-v1.x

.PHONY: all
all: $(PROGRAMS)

.SUFFIXES: .c .o .h .x

.c.o:
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

.F.o:
	$(F77) $(FFLAGS) $(CPPFLAGS) -c -o $@ $<

.o.x:
	$(LINK) $(LOADER_OPTS) $(LDFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) *.o *.x
