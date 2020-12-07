import numpy as np
from collections import Counter

class CategoricalDistribution:
    def __init__(self, column):
        n = len(column)
        c = Counter(column)
        self.categories = [k for k in c.keys()]
        self.probabilities = [v/n for v in c.values()]
        # print(self.categories)
        # print(self.probabilities)

    def sample(self, n, seed=42):
        np.random.seed(seed)
        return np.random.choice(self.categories, n, p=self.probabilities)