#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <mpi.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;
size_t nInitialSamples;

#define NSAMPLES 240
#define NPARAMETERS 2

double sampleData[NPARAMETERS + 1];
double resultData[2];

void sendSample(int &sampleId, int rank) {
  sampleData[0] = sampleId;
  getSample(sampleId, &sampleData[1]);

  MPI_Send(sampleData, NPARAMETERS + 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
  ++sampleId;
}

int receiveSample(int &numReceivedSamples) {
  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(resultData, 2, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  updateEvaluation(static_cast<int>(resultData[0]), resultData[1]);

  ++numReceivedSamples;

  return status.MPI_SOURCE;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rankId, rankCount;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

  nSamples = NSAMPLES;
  nParameters = NPARAMETERS;
  const size_t nConsumers = rankCount - 1;

  if (rankId == 0) {
    initializeSampler(nSamples, nParameters);

    // having more workers than initial samples makes no sense
    assert(1 < rankCount && rankCount < 26);

    auto t0 = std::chrono::system_clock::now();

    int sentSampleId = 0;
    int numReceivedSamples = 0;

    while (sentSampleId < nConsumers) {
      sendSample(sentSampleId, sentSampleId + 1);
    }

    while (sentSampleId < nSamples) {
      int rank = receiveSample(numReceivedSamples);
      sendSample(sentSampleId, rank);
    }

    while (numReceivedSamples < nSamples) {
      int rank = receiveSample(numReceivedSamples);
      double dummy = 0;
      MPI_Send(&dummy, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
    }

    auto t1 = std::chrono::system_clock::now();

    checkResults();
    double evalTime = std::chrono::duration<double>(t1 - t0).count();
    printf("Total Running Time: %.3fs\n", evalTime);

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
        MPI_Recv(sampleData, nParameters + 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        resultData[0] = sampleData[0];
        resultData[1] = evaluateSample(&sampleData[1]);
        MPI_Send(resultData, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      }

    } while (!done);
  }

  MPI_Finalize();

  return 0;
}
