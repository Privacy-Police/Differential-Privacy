import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme()
os.makedirs('figs/histograms/', exist_ok=True)

def make_histograms(original_data, synthetic_data, fig_name):
    # Preprocess to get rid of rows with +-inf values
    synthetic_data = synthetic_data.replace(-np.Inf, np.nan).replace(np.inf, np.nan).drop(synthetic_data.columns[0], axis=1).dropna()

    # Combine into one dataset
    original_data['Source'] = 'original'
    synthetic_data['Source'] = 'synthetic'
    combined = pd.concat([original_data, synthetic_data], axis=0)

    # Plots
    for i in range(len(combined.columns)-1):
        plt.figure(figsize=(6, 6))
        plot = sns.displot(combined, x=f"{i}", hue="Source", stat="density", common_norm=False)
        plot.savefig(f"figs/histograms/{fig_name}_{i}.png")


# ADULT
original_data = pd.read_csv('datasets/adult_subset.csv')
synthetic_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_adult.csv')
make_histograms(original_data, synthetic_data, 'ADULT')

# PUMS
original_data = pd.read_csv('datasets/pums_subset.csv')
synthetic_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_pums.csv')
make_histograms(original_data, synthetic_data, 'PUMS')

# POWER
original_data = pd.read_csv('datasets/power_subset.csv')
synthetic_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_power.csv')
make_histograms(original_data, synthetic_data, 'POWER')
