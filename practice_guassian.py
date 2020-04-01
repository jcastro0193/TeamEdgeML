import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

input_file = "data.txt"
data = np.loadtxt(input_file, delimiter=',')

X = np.column_stack([data[:-1, 1:]])
test = data[-1, 1:].reshape(1, -1)

print("X:")
print(X)
print("Test:")
print(test)

num_components = 2
model = GaussianHMM(n_components=num_components, covariance_type="diag", n_iter=1000)
model.fit(X)
print("transmat:", model.transmat_prior)

pred = model.predict(test)
print("Predictions:")
print(pred)
