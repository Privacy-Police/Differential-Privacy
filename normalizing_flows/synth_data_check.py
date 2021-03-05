"""
    This file basically checks the synthetic and original dataset.
    And compare their performance over pmse, wasserstein metric
"""
import pandas as pd
import numpy as np
from metrics import pmse
from metrics import wasserstein

power_synth = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_power.csv')
power_org = pd.read_csv('datasets/power_subset.csv')

data_org_clean = power_org.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
data_synth_clean = power_synth.replace([np.inf,-np.inf],np.nan).dropna(axis=0)
print('Cleaning done')

print(power_org.describe())
print(power_synth.describe())
print(pmse.pmse_ratio(data_synth_clean,data_org_clean))

""" Calculating wasserstein """

wass = []
for i in range(50):
  wass.append(wasserstein.wasserstein_randomization(data_org_clean, data_synth_clean, 100))
print(np.mean(wass))
print(np.std(wass))


wass2 = []
for i in range(50):
  wass.append(wasserstein.wasserstein_randomization(data_org_clean, data_org_clean, 100))
print('baseline',np.mean(wass))
print('baseline',np.std(wass))
