#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "sampler/sampler.hpp"

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

double sampleData[NPARAMETERS + 1];
double resultData[2];

void sendSample(int &sampleId, double *sampleArray, int rank) {
  sampleData[0] = sampleId;
  std::copy(&sampleArray[sampleId * NPARAMETERS],
            &sampleArray[(sampleId + 1) * NPARAMETERS], &sampleData[1]);

  MPI_Send(sampleData, NPARAMETERS + 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
  ++sampleId;
}

int receiveSample(int &numReceivedSamples, double *resultsArray) {
  MPI_Status status;
  MPI_Probe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
  MPI_Recv(resultData, 2, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  resultsArray[static_cast<int>(resultData[0])] = resultData[1];

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
    double *sampleArray = initializeSampler(nSamples, nParameters);
    double *resultsArray = (double *)calloc(nSamples, sizeof(double));
    std::vector<double> consumerTimes(nConsumers);

    auto start = std::chrono::steady_clock::now();

    int sentSampleId = 0;
    int numReceivedSamples = 0;

    while (sentSampleId < nConsumers) {
      sendSample(sentSampleId, sampleArray, sentSampleId + 1);
    }

    while (sentSampleId < nSamples) {
      int rank = receiveSample(numReceivedSamples, resultsArray);
      sendSample(sentSampleId, sampleArray, rank);
    }

    while (numReceivedSamples < nSamples) {
      int rank = receiveSample(numReceivedSamples, resultsArray);
      auto end = std::chrono::steady_clock::now();
      consumerTimes[rank - 1] =
          std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
              .count();

      double dummy = 0;
      MPI_Send(&dummy, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD);
    }

    checkResults(resultsArray);

    double totalTime =
        *std::max_element(consumerTimes.begin(), consumerTimes.end());
    double averageTime =
        std::accumulate(consumerTimes.begin(), consumerTimes.end(), 0.) /
        nConsumers;

    std::cout << "Total time:           " << totalTime << std::endl;
    std::cout << "Average time:         " << averageTime << std::endl;
    std::cout << "Load imbalance ratio: "
              << (totalTime - averageTime) / totalTime << std::endl;

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
