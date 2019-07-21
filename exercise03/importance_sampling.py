import time

import numpy as np
import scipy.stats as ss


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.interval = self.end - self.start


standard_normal = ss.norm(loc=0, scale=1)
truncated_exponential = ss.truncexpon(b=np.inf, loc=4.5, scale=1)

for N in np.logspace(4, 8, base=10, num=5, dtype=np.int):
    # estimator (3)
    with Timer() as t1:
        y = standard_normal.rvs(size=N)
        estimator_1 = np.mean(y > 4.5)

    # estimator (4)
    with Timer() as t2:
        x = truncated_exponential.rvs(size=N)
        f = standard_normal.pdf(x)
        g = truncated_exponential.pdf(x)
        estimator_2 = np.mean(f / g)

    print(f"N = {N}")
    print(f"Estimator (3):  {estimator_1:.10e} [{t1.interval:.5f}s]")
    print(f"Estimator (4):  {estimator_2:.10e} [{t2.interval:.5f}s]")
    print(f"Exact solution: {1. - standard_normal.cdf(4.5):.10e}")
    print()
