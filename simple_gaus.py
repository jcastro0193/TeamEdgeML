import numpy as np
from hmmlearn import hmm

"""This code uses an
HMM to predict the sequence of hidden
states given a sequence of observations."""

# list if [climate, icecreams]
#   climates: hot(=1), warm(=2), cold(=3)
X = [   [1, 10],
        [2, 5],
        [3, 0],
        [2, 5],
        [1, 10],
        [2, 5],
        [3, 0],
        [2, 5],
        [1, 10],
        [2, 5],
        [3, 0]]

# test list of observations
test = [[10], [5], [0], [5], [10]]

# train model
hmm = hmm.GaussianHMM(n_components=3).fit(test)

# predict hidden states based on observations
results = hmm.predict(test)

# Results should be  [1, 2, 3, 2]
print(results)

