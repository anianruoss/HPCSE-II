from functools import reduce
from math import exp, isclose, log, pi
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np
from scipy import special

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
plt.bar(bins, counts, width=0.5, align='edge', color='gray')
plt.savefig(path.join(working_dir, 'plots/hist.eps'), bbox_inches='tight')
plt.close()

counts, bins = hist(data, num_bins, normalize=True)
plt.bar(bins, counts, width=0.5, align='edge', color='gray')
plt.savefig(
    path.join(working_dir, 'plots/hist_normalized.eps'), bbox_inches='tight'
)


def poisson_likelihood(x, lambda_):
    n = x.shape[0]
    lambda_x = reduce(
        lambda y, z: y * z, (lambda_ ** x).tolist()
    )
    x_factorial = reduce(
        lambda y, z: y * z, special.factorial(x, exact=True).tolist()
    )

    return exp(- lambda_ * n) * lambda_x / x_factorial


def poisson_log_likelihood(x, lambda_):
    n = x.shape[0]
    log_lambda_x = log(lambda_) * np.sum(x)
    log_x_factorial = np.sum(np.log(special.factorial(x, exact=True)))

    return (- lambda_ * n) + log_lambda_x - log_x_factorial


# Poisson MLE
lambda_hat = np.mean(data)


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


# Gaussian MLE
mu_hat = np.mean(data)
var_hat = np.var(data)

assert isclose(
    gaussian_log_likelihood(data, mu_hat, var_hat), -40287.57, abs_tol=1e-2
)
assert np.equal(
    exp(gaussian_log_likelihood(data, mu_hat, var_hat)),
    gaussian_likelihood(data, mu_hat, var_hat)
)

print('Poisson')
print('Lambda (MLE): \t\t', lambda_hat)
try:
    print('Likelihood: \t\t', poisson_likelihood(data, lambda_hat))
except OverflowError:
    print('Likelihood: \t\t', exp(poisson_log_likelihood(data, lambda_hat)))
print('Log-Likelihood: \t', poisson_log_likelihood(data, lambda_hat))
print()

print('Gaussian')
print('Mu (MLE): \t\t', mu_hat)
print('Var (MLE): \t\t', var_hat)
print('Likelihood: \t\t', gaussian_likelihood(data, mu_hat, var_hat))
print('Log-Likelihood: \t', gaussian_log_likelihood(data, mu_hat, var_hat))

# plot gaussian distribution
data_range = np.arange(0, np.max(data), step=1)
gaussian_densities = list(
    map(
        lambda x: gaussian_likelihood(np.array([x]), mu_hat, var_hat),
        data_range
    )
)
plt.plot(data_range, gaussian_densities, c='g', label='Gaussian', linewidth=3)

# plot poisson distribution
poisson_densities = list(
    map(lambda x: poisson_likelihood(np.array([x]), lambda_hat), data_range)
)
plt.plot(data_range, poisson_densities, c='r', label='Poisson', linewidth=3)

plt.legend()
plt.savefig(
    path.join(working_dir, 'plots/hist_distributions.eps'), bbox_inches='tight'
)
