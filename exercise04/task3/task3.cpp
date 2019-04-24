#include <cassert>
#include <chrono>
#include <numeric>
#include <queue>
#include <stdio.h>
#include <upcxx/upcxx.hpp>
#include "sampler/sampler.hpp"

#define NSAMPLES 240
#define NPARAMETERS 2

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> sampleArray;

struct Consumer {
  size_t sampleId;
  double sample[NPARAMETERS];

  Consumer(size_t id) : sampleId(id) { getSample(sampleId, sample); }

  static std::pair<size_t, double> processSample(Consumer consumer) {
    return {consumer.sampleId, evaluateSample(consumer.sample)};
  }
};

void processSample(size_t consumerId, size_t &sampleId,
                   std::vector<upcxx::future<>> &futures) {
  futures[consumerId] =
      upcxx::rpc(consumerId + 1, Consumer::processSample, Consumer{sampleId++})
          .then([](std::pair<size_t, double> result) {
            updateEvaluation(result.first, result.second);
          });
}

int main(int argc, char *argv[]) {
  upcxx::init();
  rankId = upcxx::rank_me();
  rankCount = upcxx::rank_n();

  nSamples = NSAMPLES;
  nParameters = NPARAMETERS;

  if (rankId == 0) {
    printf("Processing %ld Samples (24 initially available), each with %ld "
           "Parameter(s)...\n",
           nSamples, nParameters);
    initializeSampler(nSamples, nParameters);

    // having more workers than initial samples makes no sense
    assert(1 < rankCount && rankCount < 26);

    auto t0 = std::chrono::system_clock::now();

    const size_t nConsumers = rankCount - 1;

    std::vector<upcxx::future<>> futures(nConsumers);
    size_t sampleId = 0;

    for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
      processSample(consumerId, sampleId, futures);
    }

    while (sampleId < nSamples) {
      for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
        if (futures[consumerId].ready() && sampleId < nSamples) {
          processSample(consumerId, sampleId, futures);
        } else {
          upcxx::progress();
        }
      }
    }

    upcxx::future<> conjoined_future = upcxx::make_future();

    for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
      conjoined_future = upcxx::when_all(conjoined_future, futures[consumerId]);
    }
    conjoined_future.wait();

    auto t1 = std::chrono::system_clock::now();

    checkResults();
    double evalTime = std::chrono::duration<double>(t1 - t0).count();
    printf("Total Running Time: %.3fs\n", evalTime);
  }

  upcxx::finalize();
  return 0;
}
