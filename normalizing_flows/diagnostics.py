import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import torch

def get_distribution_diagnostic_plot(model, input, model_name = 'maf', ncols = 6, nrows = 4):
  u = model(input.to('cuda'))[0].detach().cpu().numpy()

  fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(16, 10)
    )
  
  for ax in axes.reshape(-1):
    dim1 = np.random.randint(input.shape[1])  
    sns.distplot(u[:, dim1], ax=ax, color="darkorange")
    ax.set_xlabel("dim: " + str(dim1), size=14)
    ax.set_xlim(-5, 5)
  plt.savefig("figs/" + model_name + "_marginal.png", bbox_inches="tight", dpi=300)
  plt.savefig("figs/" + model_name + "_marginal.pdf", bbox_inches="tight", dpi=300)


def get_scattered_diagnostic_plot(model, input, model_name = 'maf', ncols = 6, nrows = 4):
  fig, axes = plt.subplots(
        ncols=ncols, nrows=nrows, sharex=True, sharey=True, figsize=(16, 10)
    )

  for ax in axes.reshape(-1):
        dim1 = np.random.randint(input.shape[1])
        dim2 = np.random.randint(input.shape[1])
        ax.scatter(u[:, dim1], u[:, dim2], color="dodgerblue", s=0.5)
        ax.set_ylabel("dim: " + str(dim2), size=14)
        ax.set_xlabel("dim: " + str(dim1), size=14)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_aspect(1)
  plt.savefig("figs/" + model_name + "_scatter.png", bbox_inches="tight", dpi=300)
  plt.savefig("figs/" + model_name + "_scatter.pdf", bbox_inches="tight", dpi=300)