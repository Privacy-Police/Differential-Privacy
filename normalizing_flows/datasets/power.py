import numpy as np
import matplotlib.pyplot as plt
import torch
import os

batch_size = 100

# This file is adapted from https://github.com/e-hulten/maf/blob/master/datasets/power.py

class PowerDataset:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        # Choose a file that you want to be used for power dataset
        file="datasets/power_subset.csv"
        trn, val, tst = load_data_normalised(file)

        self.train = self.Data(trn)
        self.val = self.Data(val)
        self.test = self.Data(tst)

        self.n_dims = self.train.x.shape[1]

    def show_histograms(self, split):

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError("Invalid data split")

        util.plot_hist_marginals(data_split.x)
        plt.show()
        
    def get_stats(self):
      file="datasets/power_subset.csv"
      trn, val, test = load_data(file)
      data = np.vstack((trn, val))
      mu = data.mean(axis=0)
      s = data.std(axis=0)
      return mu, s


def load_data(root_path):
    # splits data into train, validation and test sets
    data = np.genfromtxt(root_path, delimiter=',', skip_header=1)
    print('Shape of Subset Power Dataset is', data.shape)
    rng = np.random.RandomState(42)
    rng.shuffle(data)

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(root_path):
    # returns normalised train, validation and test sets
    data_train, data_validate, data_test = load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test