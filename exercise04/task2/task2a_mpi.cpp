#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rankId, rankCount;
  MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);

  nSamples = NSAMPLES;
  nParameters = NPARAMETERS;

  double *sampleArray;

  if (rankId == 0) {
    sampleArray = initializeSampler(nSamples, nParameters);
  } else {
    sampleArray = new double[nSamples * nParameters];
  }

  MPI_Bcast(sampleArray, nSamples * nParameters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double *resultsArray = (double *)calloc(nSamples, sizeof(double));

  int batchSize = nSamples / rankCount;
  int startId = rankId * batchSize;
  int endId = (rankId + 1) * batchSize;

  // for simplicity: allows easy usage of MPI_Gather
  assert(nSamples % rankCount == 0);
  if (rankId == rankCount - 1) {
    assert(endId == nSamples);
  }

  auto start = std::chrono::steady_clock::now();

  for (int localId = 0; localId < batchSize; ++localId) {
    int globalId = startId + localId;
    resultsArray[globalId] =
        evaluateSample(&sampleArray[globalId * nParameters]);
  }

  if (rankId == 0) {
    MPI_Gather(MPI_IN_PLACE, batchSize, MPI_DOUBLE, resultsArray, batchSize,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gather(resultsArray + startId, batchSize, MPI_DOUBLE, MPI_IN_PLACE,
               batchSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }

  auto end = std::chrono::steady_clock::now();

  double rankTime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();

  double maxTime, sumTime;

  MPI_Reduce(&rankTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rankTime, &sumTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rankId == 0) {
    checkResults(resultsArray);

    double totalTime = maxTime;
    double averageTime = sumTime / rankCount;

    std::cout << "Total time:           " << totalTime << std::endl;
    std::cout << "Average time:         " << averageTime << std::endl;
    std::cout << "Load imbalance ratio: "
              << (totalTime - averageTime) / totalTime << std::endl;
  }

  MPI_Finalize();

  return 0;
}
