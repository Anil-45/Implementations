"""Permutation test."""

##############################################################################
# Problem: Men spend more money compared to woman for christmas gifts
# H0 : Both spend same amount of money
# H1 : Men spend higher amount of money
# Consider significance level as 0.15
# We will use pareto dist. to generate the data as online spending is similar
# to pareto(Few people spend lots of money and most spend little amount)
##############################################################################

import numpy as np

size = 100
alpha = 0.15
men = np.random.pareto(a=5.0, size=size)
women = np.random.pareto(a=4.0, size=size)

# Making an observation
obs = abs(np.percentile(men, 50) - np.percentile(women, 50))
print("Observation: ", obs)

# Test static: abs(men_median - women_median)
combined = np.concatenate((men, women))
n_iters = 1000
test_stats = np.zeros(n_iters)

for i in range(n_iters):
    np.random.shuffle(combined)

    data1 = combined[:size]
    data2 = combined[size:]

    test_stats[i] = abs(np.percentile(data1, 50) - np.percentile(data2, 50))

# P(X >= obs | H0)
p_value = np.sum(test_stats >= obs) / n_iters

print("P-value: ", p_value)

if p_value <= alpha:
    print("Reject H0")
else:
    print("Not enough evidence to reject H0")
