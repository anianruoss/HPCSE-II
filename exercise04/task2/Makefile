CXX=upcxx
LDFLAGS  += 
CXXFLAGS += -O3 -g -std=c++17 -Wall -Wfatal-errors
BINARIES = single task2a task2b
OBJECTS = sampler/sampler.o

.SECONDARY:
.PHONY: all
all: $(BINARIES) task2a_mpi

$(BINARIES) : % : %.o $(OBJECTS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

task2a_mpi: task2a_mpi.cpp $(OBJECTS)
	mpiicpc -c task2a_mpi.cpp
	mpiicpc -o task2a_mpi task2a_mpi.o sampler/sampler.o	

.PHONY: clean
clean:
	$(RM) $(BINARIES) *.o *.ti *.optrpt *.txt task2a_mpi
