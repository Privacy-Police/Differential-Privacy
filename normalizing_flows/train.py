import argparse
import math

import numpy as np
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader

import flows as fnn
from datasets import get_datasets, get_input_size


wandb.init(project='privacy_police')
config = wandb.config

def main(args):
    # Pass config to wandb
    for key, value in vars(args).items():
       setattr(config, key, value)

    # Use CUDA GPU if available
    gpu_available = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if gpu_available else "cpu")

    # Get Dataset
    dataset = get_datasets(args.dataset_name, args.seed)
    input_size = get_input_size(args.dataset_name)

    # Make dataloader
    train, val, test = dataset
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    # Define model
    modules = []
    for _ in range(args.made_blocks):
        modules += [
            fnn.MADE(input_size, args.hidden_dims, num_cond_inputs=None, act='relu'),
            fnn.BatchNormFlow(input_size),
            fnn.Reverse(input_size)
        ]
    model = fnn.FlowSequential(*modules)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Train model
    model.train()
    for epoch_num in tqdm(range(1, args.epoch+1)):
        # Train for 1 epoch
        train_loss = 0
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Loss = Negative Log Likelihood
            loss = -model.log_probs(batch).mean()
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_loss = np.sum(train_loss) / len(train_loader)
        print(f"Epoch: {epoch_num} Average train log likelihood: {-avg_loss:.5f}")
        torch.save(model, "my_trained_maf.pt")

        # Validation
        val_loss = 0
        for batch, _ in val_loader:
            batch = batch.to(device)
            val_loss += -model.log_probs(batch).mean().item()
        avg_val_loss = np.sum(val_loss) / len(val_loader)
        print(f"Epoch: {epoch_num} Average validation log likelihood: {-avg_val_loss:.5f}")

        # Log statistics to wandb
        wandb.log({
            'epoch': epoch_num,
            'average log likelihood (train)': -avg_loss,
            'average log likelihood (validation)': -avg_val_loss
        })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizigng flows model")
    parser.add_argument('--use_cuda', default=True, type=bool, help="Whether to use GPU or CPU. True for GPU")
    parser.add_argument('--dataset_name', default='mnist', type=str, help="Dataset name to train on")
    parser.add_argument('--epoch', default=1000, type=int, help="number of epochs to train")
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training model")
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="Learning rate for the optimizer")
    parser.add_argument('--weight_decay', default=1e-6, type=float, help="Weight decay for the optimizer")
    parser.add_argument('--made_blocks', default=5, type=int, help='Number of MADE blocks for the MAF model')
    parser.add_argument('--hidden_dims', default=512, type=int, help='Number of nodes for hidden layers for each MADE block')
    args = parser.parse_args()
    main(args)
