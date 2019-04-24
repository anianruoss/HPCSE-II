#include "conduits/upcxx.h"
#include "solvers/base.h"
#include <queue>

extern Korali::Solver::Base *_solver;
extern Korali::Problem::Base *_problem;
extern double* _sampleArrayPointer;

int rankId;
int rankCount;
size_t nSamples;
size_t nParameters;
upcxx::global_ptr<double> sampleArrayPointer;

Korali::Conduit::UPCXX::UPCXX(Korali::Solver::Base *solver)
    : Base::Base(solver){};

std::queue<upcxx::future<size_t>> futures;

struct Consumer {
  double *sample = (double *)calloc(nParameters, sizeof(double));
  size_t sampleId;

  Consumer(size_t id) : sampleId(id) {
    std::copy(&sampleArrayPointer.local()[nParameters * sampleId],
              &sampleArrayPointer.local()[nParameters * (sampleId + 1)],
              sample);
  }

  static std::tuple<size_t, size_t, double> evaluateSample(Consumer consumer) {
    return {rankId, consumer.sampleId,
            _problem->evaluateSample(consumer.sample)};
  }
};

void enqueueFuture(size_t rank, size_t sampleId) {
  futures.push(upcxx::rpc(rank, Consumer::evaluateSample, Consumer{sampleId})
                   .then([](std::tuple<size_t, size_t, double> result) {
                     _solver->updateEvaluation(std::get<1>(result),
                                               std::get<2>(result));
                     return std::get<0>(result);
                   }));
}

void Korali::Conduit::UPCXX::processSample(size_t sampleId) {
  if (futures.size() < rankCount - 1) {
    enqueueFuture(sampleId + 1, sampleId);
  } else {
    upcxx::future<size_t> future = futures.front();
    futures.pop();

    upcxx::progress();

    if (future.ready()) {
      size_t rank = future.result();
      enqueueFuture(rank, sampleId);
    } else {
      futures.push(future);
    }
  }

  // wait if last sample
  if (sampleId == nSamples - 1) {
    while (!futures.empty()) {
      futures.front().wait();
      futures.pop();
    }
  }
}

void Korali::Conduit::UPCXX::run() {
  upcxx::init();
  rankId = upcxx::rank_me();
  rankCount = upcxx::rank_n();
  nSamples = _solver->_sampleCount;
  nParameters = _solver->N;

  // Creating sample array in global shared memory
  if (rankId == 0) {
    sampleArrayPointer = upcxx::new_array<double>(nSamples * nParameters);
  }
  upcxx::broadcast(&sampleArrayPointer, 1, 0).wait();

  if (rankId == 0) {
    _sampleArrayPointer = sampleArrayPointer.local();
    _solver->runSolver();
  }

  upcxx::barrier();
  upcxx::finalize();
}
