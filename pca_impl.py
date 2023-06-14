"""Implementing PCA"""

import numpy as np
from scipy import linalg
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = np.random.rand(1000, 50)
print(data.shape)

# Column standardize the data
data_ = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

########################################################
#              Scratch PCA                             #
########################################################

# Covariance matrix
# covariance = 1/n-1 * summation((x-mu_x)*(y-mu_y)), as we column standardized
# cov becomes 1/n-1 * summation((x)*(y)) and var is 1/n-1 * summation((x^2))
covariance = (1 / (data_.shape[0] - 1)) * (np.matmul(data_.T, data))

# computing eigen values, eigen vectors
eigen_val, eigen_vect = linalg.eigh(covariance)

# Note: eigen values are in ascending order
eigen_val = eigen_val[::-1]
eigen_vect = np.fliplr(eigen_vect)

# reducing dimensions of the actual data
transformed_data = np.matmul(data_, eigen_vect)
exp_var = np.sum(eigen_val[:2]) / np.sum(eigen_val)
print(f"Explained variance Imp. PCA: {exp_var}")

plt.subplot(121)
sns.scatterplot(
    x=transformed_data[:, 0], y=transformed_data[:, 1], label=exp_var
)
plt.title("Imp. PCA")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

########################################################
#                      PCA                             #
########################################################
pca = PCA(n_components=50, random_state=0)

transformed_data = pca.fit_transform(data_)
exp_var = np.sum(pca.explained_variance_[:2]) / np.sum(pca.explained_variance_)
print(f"Explained variance Act. PCA: {exp_var}")

plt.subplot(122)
sns.scatterplot(
    x=transformed_data[:, 0], y=transformed_data[:, 1], label=exp_var
)
plt.title("Act. PCA")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()
