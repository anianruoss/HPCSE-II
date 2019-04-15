#include <stdio.h>
#include <cassert>
#include <chrono>
#include <numeric>
#include <queue>
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

	Consumer(size_t id) : sampleId(id) {
		getSample(sampleId, sample);
	}

	static std::pair<size_t, double> processSample(Consumer consumer) {
		return {consumer.sampleId, evaluateSample(consumer.sample)};
	}
};

int main(int argc, char* argv[])
{
	upcxx::init();
	rankId    = upcxx::rank_me();
	rankCount = upcxx::rank_n();

	nSamples = NSAMPLES;
	nParameters = NPARAMETERS;

	if (rankId == 0) {
		printf("Processing %ld Samples (24 initially available), each with %ld Parameter(s)...\n", nSamples, nParameters); 
		initializeSampler(nSamples, nParameters);
		
		// avoid having more consumers than available samples
		assert(rankCount <= 25);

		size_t sampleId = 0;
		const size_t nConsumers = rankCount - 1;

		std::vector<upcxx::future<>> futures(rankCount);

		for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
			Consumer consumer(sampleId++);

			size_t rank = consumerId + 1;

			futures[rank] = upcxx::rpc(
				rank, Consumer::processSample, consumer
			).then([](std::pair<size_t, double> result) { 
				updateEvaluation(result.first, result.second);
					});
		}
		
		while (sampleId < nSamples) {
			for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
				if (nSamples <= sampleId) {
					break;
				}
				
				size_t rank = consumerId + 1;

				if (futures[rank].ready()) {
					Consumer consumer(sampleId++);

					futures[rank] = upcxx::rpc(
						rank, Consumer::processSample, consumer
						).then([](std::pair<size_t, double> result) { 
							updateEvaluation(result.first, result.second);
								});
				} else {
					upcxx::progress();
				}
			}	
		}

		upcxx::future<> conjoined_future = upcxx::make_future();

		for (size_t consumerId = 0; consumerId < nConsumers; ++consumerId) {
			size_t rank = consumerId + 1;
			conjoined_future = upcxx::when_all(conjoined_future, futures[rank]);
		}
		conjoined_future.wait();
		
			
		auto t0 = std::chrono::system_clock::now();

		auto t1 = std::chrono::system_clock::now();

		checkResults();
		double evalTime = std::chrono::duration<double>(t1-t0).count();
		printf("Total Running Time: %.3fs\n", evalTime);
	}

	upcxx::finalize();
	return 0;
}

