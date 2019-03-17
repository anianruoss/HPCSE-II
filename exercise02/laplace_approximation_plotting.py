import math
from os import makedirs, path

import matplotlib.pyplot as plt
import numpy as np

working_dir = path.dirname(path.abspath(__file__))
makedirs(path.join(working_dir, 'plots'), exist_ok=True)


def cauchy_pdf(x, x0_, gamma_):
    return gamma_ / (math.pi * ((x - x0_) ** 2 + gamma_ ** 2))


def laplace_approximation(x, x0_, gamma_):
    gamma_squared = gamma_ ** 2
    return np.exp(- (x - x0_) ** 2 / gamma_squared) / (math.pi * gamma_)


x0 = -2.
gamma = 1.

x_range = np.arange(-8, 4, step=0.01)
plt.plot(
    x_range,
    cauchy_pdf(x_range, x0, gamma),
    label='Cauchy Distribution',
    linewidth=3
)
plt.plot(
    x_range,
    laplace_approximation(x_range, x0, gamma),
    label='Laplace Approximation',
    linewidth=3
)
plt.xlabel('x')
plt.ylabel(r'$P\left(x\right)$')
plt.legend()
plt.savefig(
    path.join(working_dir, 'plots/laplace_approximation.eps'),
    bbox_inches='tight'
)
