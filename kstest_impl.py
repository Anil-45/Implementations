"""Implementing KS Test."""

import numpy as np
from scipy import stats

data1 = np.random.pareto(a=4.5, size=10000)
# data2 = np.random.pareto(a=3.5, size=10000)
data2 = np.random.pareto(a=4.5, size=10000)

D, p = stats.ks_2samp(data1, data2)
print(D, p)

################################################
# H0 : samples are from same distribution
# Reject H0 if
#    D > C(alpha)*((n+m)/(n*m))^(0.5)
#           where n, m are sample sizes
#
#    C(alpha) = (-0.5*ln(alpha/2))^(0.5)
#           where alpha is significance level
#
#   solving for alpha in terms of D gives P as
#
#           2*exp((-2*n*m*D*D)/(n+m))
###############################################


def ks_test(dist1: np.array, dist2: np.array):
    """KS Test.

    Args:
        dist1 (np.array): sample 1
        dist2 (np.array): sample 2

    Returns:
        float: D
        float: P
    """
    n = len(dist1)
    m = len(dist2)
    dist1 = np.sort(dist1)
    dist2 = np.sort(dist2)

    i, j = 0, 0
    temp1, temp2 = 0, 0
    d_nm = 0

    while (i < n) and (j < m):
        a = dist1[i]
        b = dist2[j]

        if a <= b:
            temp1 = (i + 1.0) / n
            i += 1
        if b <= a:
            temp2 = (j + 1.0) / m
            j += 1

        if abs(temp1 - temp2) > d_nm:
            d_nm = abs(temp1 - temp2)

    p_value = 2 * np.exp((-2 * n * m * d_nm * d_nm) / (n + m))

    return d_nm, p_value


d_imp, p_imp = ks_test(data1, data2)
print(d_imp, p_imp)

alpha = 0.05

print("Null Hypothesis : samples are from same distribution")
if p_imp <= alpha:
    print("Null Hypothesis rejected")
    print("Different Distributions")
else:
    print("Not enough evidence to reject Null Hypothesis")
