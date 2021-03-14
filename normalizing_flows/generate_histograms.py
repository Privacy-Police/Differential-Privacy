import seaborn as sns
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme()

# Read csv
data = pd.read_csv('datasets/pums_subset.csv')
synth_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_pums.csv')
synth_data = synth_data.replace(-np.Inf, np.nan).replace(np.inf, np.nan).drop(synth_data.columns[0], axis=1).dropna()

# Combine into one dataset
data['Source']='original'
synth_data['Source'] = 'synthetic'
combined = pd.concat([data, synth_data], axis=0)

# Plots
for i in range(6):
    plt.figure(figsize=(6, 6))
    plot = sns.displot(combined, x=f"{i}", hue="Source", stat="density", common_norm=False)
    plot.savefig(f"figs/histograms/adult_{i}.png")


# For PUMS
data = pd.read_csv('datasets/pums_subset.csv')
synth_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_pums.csv')
synth_data = synth_data.replace(-np.Inf, np.nan).replace(np.inf, np.nan).drop(synth_data.columns[0], axis=1).dropna()

# Combine into one dataset
data['Source']='original'
synth_data['Source'] = 'synthetic'
combined = pd.concat([data, synth_data], axis=0)

for i in range(4):
    plt.figure(figsize=(6, 6))
    plot = sns.displot(combined, x=f"{i}", hue="Source", stat="density", common_norm=False)
    plot.savefig(f"plots/pums_{i}.png")


# For POWER
data = pd.read_csv('datasets/power_subset.csv')
synth_data = pd.read_csv('synth_data/Subset_DP_Synth_Data/synth_power.csv')
synth_data = synth_data.replace(-np.Inf, np.nan).replace(np.inf, np.nan).drop(synth_data.columns[0], axis=1).dropna()


# Combine into one dataset
data['Source']='original'
synth_data['Source'] = 'synthetic'
combined = pd.concat([data, synth_data], axis=0)



data['Source']='original'
synth_data['Source'] = 'synthetic'
combined = pd.concat([data, synth_data], axis=0)
for i in range(8):
    plt.figure(figsize=(6, 6))
    plot = sns.displot(combined, x=f"{i}", hue="Source", stat="density", common_norm=False)
    plot.savefig(f"plots/power_{i}.png")




