CXX=upcxx
LDFLAGS  += 
CXXFLAGS += -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = single task3 
OBJECTS = sampler/sampler.o

.SECONDARY:
.PHONY: all
all: $(BINARIES) task3_mpi

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

task3_mpi: task3_mpi.cpp $(OBJECTS)
	mpiicpc -c task3_mpi.cpp
	mpiicpc -o task3_mpi task3_mpi.o sampler/sampler.o	

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt task3_mpi
