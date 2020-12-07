import numpy as np

class GaussianDistribution:
    def __init__(self, column):
        self.mean = np.mean(column)
        self.std = np.std(column)
        # print('mean:', self.mean)
        # print('std:', self.std)

    def sample(self, n, seed=42):
        np.random.seed(seed)
        return np.random.normal(loc=self.mean, scale=self.std, size=n)