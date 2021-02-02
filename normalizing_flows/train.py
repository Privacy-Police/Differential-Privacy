import argparse
import math

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from maf import MAF
from datasets import get_datasets, get_input_size

def debug(tensor):
    print('printing tensor')
    print(type(tensor))
    print(tensor.shape)

def main(args):
    # Get Dataset
    dataset = get_datasets(args.dataset_name, args.seed)
    input_size = get_input_size(args.dataset_name)

    # Make dataloader
    train, val, test = dataset
    train_loader = DataLoader(train, batch_size=args.batch_size)
    val_loader = DataLoader(val, batch_size=args.batch_size)
    test_loader = DataLoader(test, batch_size=args.batch_size)

    # Define model
    model = MAF(input_size, args.made_blocks, args.hidden_dims)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    

    # Train model
    for epoch_num in tqdm(range(1, args.epoch+1)):
        model.train()
        train_loss = 0
        for batch, _ in train_loader:
            #print("batch.shape",batch.shape # [128, 784]
            # Forward pass
            u, log_det = model.forward(batch)
            #print('u, log_det', u.shape, log_det.shape) # torch.Size([128, 784]) / torch.Size([128])

            # Loss = Negative Log Likelihood
            loss = 0.5 * (u ** 2).sum(dim=1)
            loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            loss -= log_det
            loss = torch.mean(loss)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = np.sum(train_loss) / len(train_loader)
        print(f"Epoch: {epoch_num} Average loss: {avg_loss:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizigng flows model")
    parser.add_argument('--dataset_name', default='mnist', type=str, help="Dataset name to train on")
    parser.add_argument('--epoch', default=1000, type=int, help="number of epochs to train")
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training model")
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="Learning rate for the optimizer")
    parser.add_argument('--weight_decay', default=1e-6, type=float, help="Weight decay for the optimizer")
    parser.add_argument('--made_blocks', default=5, type=int, help='Number of MADE blocks for the MAF model')
    parser.add_argument('--hidden_dims', default=[512], nargs="+", help='Number of nodes for hidden layers for each MADE block')
    args = parser.parse_args()
    main(args)
