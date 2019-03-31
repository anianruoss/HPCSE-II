import time

import numpy as np
import scipy.stats as ss

np.random.seed(0)

n = 10 ** 4

standard_normal = ss.norm(loc=0, scale=1)
truncated_exponential = ss.truncexpon(b=np.inf, loc=4.5, scale=1)

# estimator (3)
start_1 = time.perf_counter()

y = standard_normal.rvs(size=n)

estimator_1 = (4.5 < y).sum() / n

end_1 = time.perf_counter()

# estimator (3)
start_2 = time.perf_counter()

x = truncated_exponential.rvs(size=n)
f = np.array(list(map(standard_normal.pdf, x)))
g = np.array(list(map(truncated_exponential.pdf, x)))

estimator_2 = (f / g)[4.5 < x].sum() / n

end_2 = time.perf_counter()

print(
    'Estimator (3):   {:3.10f} [{:2.5f}s]'.format(estimator_1, end_1 - start_1)
)
print(
    'Estimator (4):   {:3.10f} [{:2.5f}s]'.format(estimator_2, end_2 - start_2)
)
print(
    'Exact solution:  {:3.10f}'.format(1. - standard_normal.cdf(4.5))
)
