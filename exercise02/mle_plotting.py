from math import exp, gamma, isclose, log, pi
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

working_dir = path.dirname(path.abspath(__file__))
makedirs(path.join(working_dir, 'plots'), exist_ok=True)

try:
    data = np.load(path.join(working_dir, 'data.npy'))
except FileNotFoundError:
    data = np.load(path.join(working_dir, 'task4.npy'))


def hist(x_array, n_bins, normalize=True):
    min_val = x_array.min()
    max_val = x_array.max()
    count = np.zeros(int(n_bins))

    for x in x_array:
        bin_number = int((n_bins - 1) * ((x - min_val) / (max_val - min_val)))
        count[bin_number] += 1

    # normalize the distribution
    if normalize:
        count /= x_array.shape[0]

    return count, np.linspace(min_val, max_val, num=n_bins)


num_bins = 100
counts, bins = hist(data, num_bins, normalize=False)
plt.bar(bins, counts, width=0.5, align='edge')
plt.savefig(path.join(working_dir, 'plots/hist.eps'), bbox_inches='tight')
plt.close()

counts, bins = hist(data, num_bins, normalize=True)
plt.bar(bins, counts, width=0.5, align='edge')
plt.savefig(
    path.join(working_dir, 'plots/hist_normalized.eps'), bbox_inches='tight'
)


def beta_likelihood(x, alpha, beta):
    b_alpha_beta = gamma(alpha) * gamma(beta) / gamma(alpha + beta)
    n = x.shape[0]

    x_alpha = np.prod(x ** (alpha - 1))
    x_beta = np.prod((1. - x) ** (beta - 1))

    return (x_alpha * x_beta) / (b_alpha_beta ** n)


def beta_log_likelihood(x, alpha, beta):
    log_b_alpha_beta = log(gamma(alpha) * gamma(beta) / gamma(alpha + beta))
    n = x.shape[0]

    log_x_alpha = (alpha - 1) * np.sum(log(x))
    log_x_beta = (beta - 1) * np.sum(log(1. - x))

    return log_x_alpha + log_x_beta - n * log_b_alpha_beta


# TODO: Subtask 6. - Calculate MLE(s) of the params from the data


def gaussian_likelihood(x, mu, var):
    n = x.shape[0]
    normalization_factor = (2. * pi * var) ** (-.5 * n)

    x_minus_mu_squared = np.sum((x - mu) ** 2)

    return normalization_factor * exp(- x_minus_mu_squared / (2. * var))


def gaussian_log_likelihood(x, mu, var):
    n = x.shape[0]
    log_normalization_factor = (-.5 * n) * log(2. * pi * var)

    x_minus_mu_squared = np.sum((x - mu) ** 2)

    return log_normalization_factor - x_minus_mu_squared / (2. * var)


# gaussian MLE
mu_hat = np.mean(data)
var_hat = np.var(data)

assert isclose(
    gaussian_log_likelihood(data, mu_hat, var_hat), -40287.57, abs_tol=1e-2
)
assert np.equal(
    exp(gaussian_log_likelihood(data, mu_hat, var_hat)),
    gaussian_likelihood(data, mu_hat, var_hat)
)

# TODO : Subtask 7. - Uncomment and evaluate likelihoods
"""
assert np.equal(
    exp(beta_log_likelihood()),
    beta_likelihood()
)
"""

# TODO : Subtask 8. - Plot the density of your distribution
# plot gaussian distribution
data_range = np.arange(0, np.max(data), step=0.1)
gaussian_density = list(
    map(
        lambda x: gaussian_likelihood(np.array([x]), mu_hat, var_hat),
        data_range
    )
)
plt.plot(data_range, gaussian_density, c='k', label='Gaussian')

# plot beta distribution
alpha, beta, loc, scale = stats.beta.fit(data)
plt.plot(
    data_range,
    stats.beta.pdf(data_range, alpha, beta, loc=loc, scale=scale),
    c='r',
    label='Beta'
)

plt.legend()
plt.savefig(
    path.join(working_dir, 'plots/hist_distributions.eps'), bbox_inches='tight'
)
