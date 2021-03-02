import numpy as np
import pandas as pd
# from datasets import adult
# from datasets import pums
# from datasets import power

## Setting up fixed random number generator
rng = np.random.RandomState(42)

"""
    For Adult Dataset
    In the following lines, we are subsetting Adult dataset to 3k rows
"""
# Getting Adult Dataset
adult = np.genfromtxt('datasets/adult_cleaned.csv', delimiter=',', skip_header=1)
adult = pd.DataFrame(adult)
print(adult.shape)
## Sampling 3k rows with fixed seed
adult_sf = adult.sample(n=3000,replace=False,random_state=42)
adult_sf.to_csv('datasets/adult_subset.csv')
print(adult_sf.shape)
print('Adult Dataset is Subset')


"""
    For PUMS Dataset
    In the following lines, we are subseting Adult dataset to 3k rows
"""
# Getting Adult Dataset
pums = np.genfromtxt('datasets/pums_cleaned.csv', delimiter=',', skip_header=1)
pums = pd.DataFrame(pums)
print(pums.shape)
## Sampling 3k rows with fixed seed
pums_sf = pums.sample(n=3000, replace=False, random_state=42)
pums_sf.to_csv('datasets/pums_subset.csv')
print(pums_sf.shape)
print('PUMS Dataset is Subsetted')


"""
    For POWER Dataset
    In the following lines, we are subseting Adult dataset to 3k rows
"""
# Getting Adult Dataset
power = np.load('datasets/power_cleaned.npy')
power = pd.DataFrame(power)
print(power.shape)
## Sampling 3k rows with fixed seed
power_sf = power.sample(n=3000, replace=False, random_state=42)
power_sf.to_csv('datasets/power_subset.csv')
print(power_sf.shape)
print('POWER Dataset is Subsetted')









