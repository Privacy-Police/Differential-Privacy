import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

import warnings
warnings.filterwarnings("ignore")

gpu_available = torch.cuda.is_available()
device = "cuda" if gpu_available else "cpu"

# Function to plot the distributions for the base density 'u' of the input dataset
def get_distribution_diagnostic_plot(model_path, input, data_name = 'MNIST', model_name = 'maf', ncols = 6, nrows = 1):
  model = torch.load(model_path, map_location=torch.device(device))

  model.to(device)
  model.eval()
  ncols = input.shape[1]
  
  #Obtain u by backpropagation
  u = model(input.to(device))[0].detach().cpu().numpy()
  
  fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(16, 8)
    )
  
  i=0
  for ax in axes.reshape(-1):
    dim1 = i
    i=i+1
    sns.distplot(u[:, dim1], ax=ax, color="darkorange")
    ax.set_xlabel("dim: " + str(dim1), size=14)
    ax.set_xlim(-5, 5)
  fig.suptitle("Distribution plot for " + data_name + " Dataset", fontsize = 13)
  outpath = "figs/" + model_name + "_" + data_name + "_marginal"
  plt.savefig(outpath + ".png", dpi=300)
  #plt.savefig(outpath + ".pdf", dpi=300)
  print("Distribution plots saved to ", outpath)

# Function to plot the pairwise scatterplot for the base density 'u' of the input dataset
def get_scattered_diagnostic_plot(model_path, input, data_name = 'MNIST', model_name = 'maf', ncols = 6, nrows = 6):
  model = torch.load(model_path, map_location=torch.device(device))
  model.to(device)
  model.eval()
  
  #Obtain u by backpropagation
  u = model(input.to(device))[0].detach().cpu().numpy()
  ncols = input.shape[1]
  nrows = input.shape[1]
  fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(16, 10)
    )

  i=0
  for ax in axes.reshape(-1):
        dim1 = int(i%ncols)
        dim2 = int(i/ncols)  
        i=i+1
        ax.scatter(u[:, dim1], u[:, dim2], color="dodgerblue", s=0.5)
        ax.set_ylabel("dim: " + str(dim2), size=14)
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect(1)
  fig.suptitle("Scatter plot for " + data_name + " Dataset", fontsize = 13)
  plt.savefig("figs/" + model_name + "_" + data_name + "_scatter.png", dpi=300)
  outpath = "figs/" + model_name + "_" + data_name + "_scatter"
  plt.savefig(outpath + ".png", dpi=300)
  #plt.savefig(outpath + ".pdf", dpi=300)
  print("Scatter plot saved to ", outpath)


def print_configs(args, wandb):
    print('Printing configuration...')
    print('---------------------------------')
    for key, value in vars(args).items():
        setattr(wandb.config, key, value)
        print(f'{key:25}: {value}')
    print('---------------------------------')
