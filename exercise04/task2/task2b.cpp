#include "sampler/sampler.hpp"
#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdio.h>
#include <upcxx/upcxx.hpp>
#include <vector>

#define NSAMPLES 240
#define NPARAMETERS 2

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;

struct Consumer {
  size_t rankId;
  size_t sampleId;
  upcxx::future<double> future;
};

int main(int argc, char *argv[]) {
  upcxx::init();
  rankId = upcxx::rank_me();
  rankCount = upcxx::rank_n();

  nSamples = NSAMPLES;
  nParameters = NPARAMETERS;

  double *sampleArray = initializeSampler(nSamples, nParameters);
  double *resultsArray = (double *)calloc(nSamples, sizeof(double));

  if (rankId == 0) {
    printf("Processing %ld Samples each with %ld Parameter(s)...\n", nSamples,
           nParameters);
  }

  auto start = std::chrono::steady_clock::now();

  if (rankId == 0) {
    std::queue<Consumer> consumers;

    const size_t nConsumers = rankCount - 1;
    std::vector<double> consumerTimes(nConsumers);
    std::vector<size_t> samples(nSamples);
    std::iota(std::begin(samples), std::end(samples), 0);

    for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
      size_t rank = consumerId + 1;
      size_t sample = samples.back();
      samples.pop_back();

      upcxx::future<double> future = upcxx::rpc(
          rank,
          [&sampleArray](size_t sampleId) -> double {
            return evaluateSample(&sampleArray[sampleId * nParameters]);
          },
          sample);
      consumers.emplace(Consumer{rank, sample, future});
    }

    while (!consumers.empty()) {
      Consumer consumer = consumers.front();
      consumers.pop();

      if (consumer.future.ready()) {
        resultsArray[consumer.sampleId] = consumer.future.result();

        if (!samples.empty()) {
          size_t sample = samples.back();
          samples.pop_back();

          upcxx::future<double> future = upcxx::rpc(
              consumer.rankId,
              [&sampleArray](size_t sampleId) -> double {
                return evaluateSample(&sampleArray[sampleId * nParameters]);
              },
              sample);
          consumer.future = future;
          consumer.sampleId = sample;
          consumers.push(consumer);
        } else {
          auto end = std::chrono::steady_clock::now();
          consumerTimes[consumer.rankId - 1] =
              std::chrono::duration_cast<std::chrono::duration<double>>(end -
                                                                        start)
                  .count();
        }

      } else {
        consumers.push(consumer);
        upcxx::progress();
      }
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
  }

  upcxx::finalize();
  return 0;
}
