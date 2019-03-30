import math
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

np.random.seed(0)


class CoinFlip:
    """
    Defines the posterior distribution for random coin flips
    """

    def __init__(self, num_tosses, num_heads):
        self.num_tosses = num_tosses
        self.num_heads = num_heads

    def eval(self, prob_head):
        uniform_prior = 1. if (0 <= prob_head <= 1) else 0.
        binomial_likelihood = (
                (prob_head ** self.num_heads) *
                ((1. - prob_head) ** (self.num_tosses - self.num_heads))
        )

        return binomial_likelihood * uniform_prior


class LogCoinFlip:
    """
    Defines the logarithmic posterior distribution for random coin flips
    """

    def __init__(self, num_tosses, num_heads):
        self.num_tosses = num_tosses
        self.num_heads = num_heads

    def eval(self, prob_head):
        if not 0. <= prob_head <= 1.:
            return -np.inf
        else:
            log_uniform_prior = 0.
            log_binomial_likelihood = (
                    self.num_heads * math.log(prob_head) +
                    (self.num_tosses - self.num_heads) *
                    math.log(1. - prob_head)
            )

            return log_binomial_likelihood + log_uniform_prior


def mcmc(target, start, iterations=10 ** 6, burn_in=10 ** 4):
    """
    Markov Chain Monte Carlo (MCMC) - Metropolis Hastings
    """

    if iterations <= burn_in:
        raise ValueError(
            f'Number of samples {iterations} smaller than burn-in {burn_in}'
        )

    current_sample = start

    # proposal distribution is a Gaussian with mean 0.0 and std 0.1
    proposal_dist = ss.norm(loc=0.0, scale=0.1)
    uniform_dist = ss.uniform()

    samples = np.empty(iterations + 1)
    samples[0] = start

    for itr in range(iterations):
        # sample candidate from proposal distribution
        sample_candidate = current_sample + proposal_dist.rvs()

        # compute metropolis acceptance probability
        acceptance_prob = min(
            1.0,
            target.eval(sample_candidate) / target.eval(current_sample)
        )

        # accept or reject with uniform probability
        temp = uniform_dist.rvs()

        # check whether to accept the candidate
        if temp <= acceptance_prob:
            current_sample = sample_candidate

        samples[itr + 1] = current_sample

    # keep only samples after burn-in iterations
    return samples[burn_in:]


def mcmc_log(log_target, start, iterations=10 ** 6, burn_in=10 ** 4):
    """
    Markov Chain Monte Carlo (MCMC) - Metropolis Hastings
    """

    if iterations <= burn_in:
        raise ValueError(
            f'Number of samples {iterations} smaller than burn-in {burn_in}'
        )

    curr_sample = start

    # proposal distribution is a Gaussian with mean 0.0 and std 0.1
    proposal_dist = ss.norm(loc=0.0, scale=0.1)
    uniform_dist = ss.uniform()

    samples = np.empty(iterations + 1)
    samples[0] = start

    for itr in range(iterations):
        # sample candidate from proposal distribution
        sample_candidate = curr_sample + proposal_dist.rvs()

        # compute log of metropolis acceptance probability
        log_acceptance_prob = min(
            0.,
            log_target.eval(sample_candidate) - log_target.eval(curr_sample)
        )

        # accept or reject with uniform probability
        temp = uniform_dist.rvs()

        # check whether to accept the candidate
        if math.log(temp) <= log_acceptance_prob:
            curr_sample = sample_candidate

        samples[itr + 1] = curr_sample

    # keep only samples after burn-in iterations
    return samples[burn_in:]


burn_in_iterations = 10 ** 2
num_iterations = 10 ** 5
starting_sample = 0.5

fig, ax = plt.subplots(2, 2, sharex='all')

for i, (tosses, heads) in enumerate([(300, 150), (3000, 1500)]):
    samples_mcmc = mcmc(
        CoinFlip(tosses, heads),
        starting_sample,
        num_iterations,
        burn_in_iterations
    )

    samples_mcmc_log = mcmc_log(
        LogCoinFlip(tosses, heads),
        starting_sample,
        num_iterations,
        burn_in_iterations
    )

    ax[0, i].hist(
        samples_mcmc, density=True, facecolor='g', alpha=0.6, label='MCMC'
    )
    ax[0, i].set_title(f'[{heads} heads / {tosses} tosses]')
    ax[0, i].legend()

    ax[1, i].hist(
        samples_mcmc_log, density=True, facecolor='b', alpha=0.6,
        label='MCMC-LOG'
    )
    ax[1, i].set_xlim([0, 1])
    ax[1, i].legend()

makedirs('plots', exist_ok=True)

plt.savefig(
    path.join(path.dirname(__file__), 'plots/coin_toss.eps'),
    bbox_inches='tight'
)
