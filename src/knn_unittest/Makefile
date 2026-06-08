include ../../config.h
include ../cpptrajfiles

SAN= -fsanitize=address -fno-omit-frame-pointer

CXXFLAGS += $(SAN) -g -O1
LDFLAGS  += -fsanitize=address

##will be (re-)building pretty much all of cpptraj, but from a different path:
VPATH += ../ ../Cluster ../Structure ../Energy
CPPTRAJ_INC += -I.. -I../tng -I../xdrfile -I../arpack -I../Cluster
##

# General rules, cut pasted from main makefile
.cpp.o:
	$(VB)echo CXX $<
	$(CXX) $(DIRECTIVES) $(CPPTRAJ_INC) $(CXXFLAGS) -c -o $@ $<

.c.o:
	$(VB)echo CC $<
	$(CC) $(DIRECTIVES) $(CPPTRAJ_INC) $(CFLAGS) -c -o $@ $<


LIBCPPTRAJ=../lib/


##cpptraj build line in src/Makefile : $(OBJECTS) $(CUDA_TARGET) $(FFT_TARGET) $(READLINE_LIB) $(CPPTRAJ_LIB) $(LDFLAGS)

##copy some required definitions from the primary makefile:

include ../Cluster/clusterfiles
CLUSTER_SOURCEFILES=$(CLUSTER_SOURCES:%.cpp=Cluster/%.cpp)
include ../Structure/structurefiles
STRUCTURE_SOURCEFILES=$(STRUCTURE_SOURCES:%.cpp=Structure/%.cpp)
include ../Energy/energyfiles
ENERGY_SOURCEFILES=$(ENERGY_SOURCES:%.cpp=Energy/%.cpp)


# All object files except the cpptraj main.cpp: replace this in order to test specific functions in isolation.
SOURCES_NO_MAIN := $(filter-out main.cpp, $(SOURCES))
OBJECTS=$(SOURCES_NO_MAIN:.cpp=.o) $(CSOURCES:.c=.o) $(CLUSTER_SOURCEFILES:.cpp=.o) $(STRUCTURE_SOURCEFILES:.cpp=.o) $(ENERGY_SOURCEFILES:.cpp=.o)

all: knn_unittest


knn_unittest.o: knn_unittest.cpp
	$(CXX) $(DIRECTIVES) $(CPPTRAJ_INC) $(CXXFLAGS) -c $< -o $@

cpptraj_objs:
	$(MAKE) -C .. CXXFLAGS="$(CXXFLAGS) $(SAN)" $(OBJECTS)

knn_unittest:  knn_unittest.o cpptraj_objs
	cd .. && $(CXX) -o knn_unittest/knntest $(OBJECTS) knn_unittest/knn_unittest.o $(CUDA_TARGET) $(FFT_TARGET) $(READLINE_LIB) $(CPPTRAJ_LIB) $(LDFLAGS)




.PHONY: clean
clean:
	rm -f knn_unittest.o

.PHONY: clobber
clobber: clean
	make -C ../ clean


