#include "sampler/sampler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdio.h>
#include <upcxx/upcxx.hpp>

size_t nSamples;
size_t nParameters;

#define NSAMPLES 240
#define NPARAMETERS 2

int main(int argc, char *argv[]) {
  upcxx::init();
  int rankId = upcxx::rank_me();
  int rankCount = upcxx::rank_n();

  nSamples = NSAMPLES;
  nParameters = NPARAMETERS;

  double *sampleArray = initializeSampler(nSamples, nParameters);

  int batchSize = std::ceil(static_cast<float>(nSamples) / rankCount);
  int startId = rankId * batchSize;
  int endId = std::min((rankId + 1) * batchSize, static_cast<int>(nSamples));

  auto start = std::chrono::steady_clock::now();

  upcxx::global_ptr<double> gptr;

  if (rankId == 0) {
    gptr = upcxx::new_array<double>(nSamples);
  }

  upcxx::broadcast(&gptr, 1, 0).wait();

  upcxx::future<> futures = upcxx::make_future();

  for (int localId = 0; localId < endId - startId; ++localId) {
    int globalId = startId + localId;
    double result = evaluateSample(&sampleArray[globalId * nParameters]);
    auto future = upcxx::rput(&result, gptr + globalId, 1);
    futures = upcxx::when_all(futures, future);
  }

  futures.wait();

  auto end = std::chrono::steady_clock::now();

  double rankTime =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
          .count();

  auto sumTime = upcxx::reduce_one(rankTime, upcxx::op_fast_add, 0);
  auto maxTime = upcxx::reduce_one(rankTime, upcxx::op_fast_max, 0);

  upcxx::barrier();

  if (rankId == 0) {
    checkResults(gptr.local());

    sumTime.wait();
    maxTime.wait();

    double totalTime = maxTime.result();
    double averageTime = sumTime.result() / rankCount;

    std::cout << "Total time:           " << totalTime << std::endl;
    std::cout << "Average time:         " << averageTime << std::endl;
    std::cout << "Load imbalance ratio: "
              << (totalTime - averageTime) / totalTime << std::endl;
  }

  upcxx::finalize();

  return 0;
}
