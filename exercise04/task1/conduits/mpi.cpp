#include "conduits/single.h"
#include "solvers/base.h"
#include <cassert>
#include <mpi.h>

extern Korali::Solver::Base *_solver;
extern Korali::Problem::Base *_problem;

int _rankId;
int _rankCount;
size_t _nSamples;
size_t _nParameters;
double *_sampleArrayPointer;

Korali::Conduit::Single::Single(Korali::Solver::Base *solver)
    : Base::Base(solver){};

double *sampleData;
double *resultData;
int sentSampleId;
int numReceivedSamples;

void sendSample(int &sampleId, int rank) {
  sampleData[0] = sampleId;
  std::copy(&_sampleArrayPointer[sampleId * _nParameters],
            &_sampleArrayPointer[(sampleId + 1) * _nParameters],
            &sampleData[1]);

  MPI_Send(sampleData, _nParameters + 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
  ++sampleId;
}

int receiveSample(int &numReceivedSamples) {
  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(resultData, 2, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  _solver->updateEvaluation(static_cast<int>(resultData[0]), resultData[1]);

  ++numReceivedSamples;

  return status.MPI_SOURCE;
}

void Korali::Conduit::Single::processSample(size_t sampleId) {
  if (sampleId == 0) {
    sentSampleId = 0;
    numReceivedSamples = 0;
  }

  if (sentSampleId < _rankCount - 1) {
    sendSample(sentSampleId, sentSampleId + 1);
  } else if (sentSampleId < _nSamples) {
    int rank = receiveSample(numReceivedSamples);
    sendSample(sentSampleId, rank);
  }

  if (sampleId == _nSamples - 1) {
    while (numReceivedSamples < _nSamples) {
      int rank = receiveSample(numReceivedSamples);
    }
  }
}

void Korali::Conduit::Single::run() {
  MPI_Init(nullptr, nullptr);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &_rankCount);

  _nSamples = _solver->_sampleCount;
  _nParameters = _solver->N;
  sampleData = (double *)calloc(_nParameters + 1, sizeof(double));
  resultData = (double *)calloc(2, sizeof(double));

  if (_rankId == 0) {
    _sampleArrayPointer =
        (double *)calloc(_nSamples * _nParameters, sizeof(double));
    _solver->runSolver();

    for (int rank = 1; rank < _rankCount; ++rank) {
      double dummy = 0;
      MPI_Send(&dummy, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
    }

  } else {
    bool done = false;

    do {
      int numElements;
      MPI_Status status;

      MPI_Probe(0, 0, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &numElements);

      if (numElements == 1) {
        done = true;
      } else {
        MPI_Recv(sampleData, _nParameters + 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        resultData[0] = sampleData[0];
        resultData[1] = _problem->evaluateSample(&sampleData[1]);
        MPI_Send(resultData, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }

    } while (!done);
  }

  MPI_Finalize();
}
