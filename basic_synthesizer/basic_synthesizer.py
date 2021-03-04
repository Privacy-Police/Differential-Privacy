import pandas as pd

from categorical import CategoricalDistribution
from gaussian import GaussianDistribution

class BasicSynthesizer:
    """
    Basic Synthesizer to compare with Normalizing Flows model
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.analyze_dataset()
    
    def analyze_dataset(self):
        self.columns = {} # column_name : Distribution
        for column_name, _ in self.dataset.iteritems():
            column = self.dataset[column_name]
            if str(column.dtype) == 'int64' or str(column.dtype) == 'float64':
                self.columns[column_name] = GaussianDistribution(column)
            elif str(column.dtype) == 'object':
                self.columns[column_name] = CategoricalDistribution(column)
            else:
                print('Unrecognized column type!')
    
    def sample(self, n):
        generated_columns = [pd.Series(dist.sample(n), name=column_name) for column_name, dist in self.columns.items()]
        return pd.concat(generated_columns, axis=1)


