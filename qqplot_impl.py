"""Implementing QQ-Plot."""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

data = np.random.exponential(scale=2.0, size=10000)
data, _ = stats.boxcox(data)
norm = np.random.normal(loc=0, scale=1, size=10000)

data_percentiles = []
norm_percentiles = []

delta = 0.01
for i in np.arange(0, 100 + delta, delta):
    data_percentiles.append(np.percentile(data, i))
    norm_percentiles.append(np.percentile(norm, i))

data_percentiles = np.array(data_percentiles)
norm_percentiles = np.array(norm_percentiles)

plt.subplot(121)
plt.plot(norm_percentiles, data_percentiles, "bo")
m, b = np.polyfit(norm_percentiles, data_percentiles, 1)
plt.plot(norm_percentiles, m * norm_percentiles + b, color="red")
plt.xlabel("Theoretical quantiles")
plt.ylabel("Sample quantiles")
plt.title("Imp. QQ")

plt.subplot(122)
stats.probplot(data, dist="norm", plot=plt)
plt.title("Act. QQ")
plt.show()
