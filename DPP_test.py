"""
This file tests k-DPP functionality as implemented in dpp_mbd
"""

import numpy as np
import matplotlib.pyplot as plt
from src import dpp_mbd

# Set up a ground set of points
N = 30
dim = 2
sigma = 15

Y = np.zeros(shape=(N**2, dim))
index = 0

# You can use a numpy mgrid for this!
for i in range(0, N):
    for j in range(0, N):
        Y[index, :] = [i, j]
        index += 1

# Set up the L-kernel
L = np.zeros(shape=(N**2, N**2))
for i in range(0, N**2):
    for j in range(0, N**2):
        
        L[i][j] = np.exp(-1/(2 * sigma) * np.linalg.norm(Y[i][:] - Y[j][:])**2)

# Now generate a sample

# Make a DPP object using the KDPP class
DPP = dpp_mbd.KDPP(L, 20)

# Gen sample and print it
dpp_sample = DPP.sample_exact_k()
print(dpp_sample)
print(Y[dpp_sample, 0], ", ", Y[dpp_sample, 1])

# Uniform for comparison
unif_smpl = np.random.permutation(len(Y))[:len(dpp_sample)]

# Plot the results
plt.title("dpp")
plt.scatter(Y[dpp_sample, 0],Y[dpp_sample, 1])
plt.show()

plt.title("unif")
plt.scatter(Y[unif_smpl, 0], Y[unif_smpl, 1])
plt.show()