import argparse
import math

import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader
from opacus import PrivacyEngine

import flows as fnn
from dataset_loader import get_datasets, get_input_size
import patch_opacus


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
            #fnn.BatchNormFlow(input_size),
            fnn.Reverse(input_size)
        ]
    model = fnn.FlowSequential(*modules)
    model.to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.enable_dp:
      privacy_engine = PrivacyEngine(
        model,
        batch_size = args.batch_size,
        sample_size = len(train_loader.dataset),
        alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
        noise_multiplier = args.sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        secure_rng=args.secure_rng,
      )
      privacy_engine.attach(optimizer)
    
    # Train model
    best_validation_loss = float('inf')
    consecutive_bad_count = 0
    model.train()
    for epoch_num in range(1, args.epoch+1):
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

        # Validation
        val_loss = 0
        for batch, _ in val_loader:
            batch = batch.to(device)
            val_loss += -model.log_probs(batch).mean().item()
        avg_val_loss = np.sum(val_loss) / len(val_loader)

        if args.enable_dp:
          epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        else:
          epsilon, best_alpha = None
        # Log statistics to wandb and stdout
        description = f'Epoch {epoch_num:3} | train LL: {-avg_loss:12.5f} | val LL: {-avg_val_loss:12.5f} | epsilon: {-epsilon:12.5f} | best alpha: {-best_alpha:12.5f}'
        print(description)
        wandb.log({
            'epoch': epoch_num,
            'average log likelihood in nats (train)': -avg_loss,
            'average log likelihood in nats (validation)': -avg_val_loss,
            'epsilon': epsilon,
            'best alpha': best_alpha
        })

        # Early stopping
        if avg_val_loss < best_validation_loss:
            best_validation_loss = avg_val_loss
            consecutive_bad_count = 0
            torch.save(model, "my_trained_maf.pt") # Save best model
        else:
            consecutive_bad_count += 1
        if consecutive_bad_count >= args.patience:
            print(f'No improvement for {args.patience} epochs. Early stopping...')
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizigng flows model")
    parser.add_argument('--patience', default=30, type=int, help="How many epochs to tolerate for early stopping")
    parser.add_argument('--use_cuda', default=True, type=bool, help="Whether to use GPU or CPU. True for GPU")
    parser.add_argument('--dataset_name', default='mnist', type=str, help="Dataset name to train on")
    parser.add_argument('--epoch', default=1000, type=int, help="number of epochs to train")
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', default=128, type=int, help="Batch size for training model")
    parser.add_argument('--learning_rate', default=1e-4, type=float, help="Learning rate for the optimizer")
    parser.add_argument('--weight_decay', default=1e-6, type=float, help="Weight decay for the optimizer")
    parser.add_argument('--made_blocks', default=5, type=int, help='Number of MADE blocks for the MAF model')
    parser.add_argument('--hidden_dims', default=512, type=int, help='Number of nodes for hidden layers for each MADE block')
    parser.add_argument('--enable_dp', default=True, type=bool, help='Whether to train model with Differential Privacy (DP) constraints. True for DP')
    parser.add_argument('--sigma', default=1.0, type=float, help='Noise multiplier (default 1.0)')
    parser.add_argument('----max-per-sample-grad_norm', default=1.0, type=float, help='Clip per-sample gradients to this norm (default 1.0)')
    parser.add_argument('--secure_rng', default=False, type=bool, help='Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost')
    parser.add_argument('--delta', default=1e-5, type=float, help="Target delta (default: 1e-5)")
    args = parser.parse_args()
    main(args)
