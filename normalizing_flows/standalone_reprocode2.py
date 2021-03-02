import argparse
import math
import time
import numpy as np
import scipy.linalg
import scipy as sp
import wandb
import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
import torch.nn as nn
import torch.nn.functional as F

# import flows as fnn
# from dataset_loader import get_datasets, get_input_size
# import patch_opacus

wandb.init(project='privacy_police')
config = wandb.config

print('Loading the dataset')


def get_mnist_datasets(random_seed, alpha=1e-6):
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: alpha + (1 - 2 * alpha) * x),
        transforms.Lambda(lambda x: torch.log(x / (1.0 - x))),
        transforms.Lambda(lambda x: torch.flatten(x))
        ])
    train_val = MNIST('../data', train=True, download=True, transform=transform)
    test = MNIST('../data', train=False, download=True, transform=transform)

    train, val = random_split(train_val, [50000, 10000], generator=torch.Generator().manual_seed(random_seed))

    # Returns 50K / 10K / 10K sized datasets
    return train, val, test


class AdultDataset:
    class Data:
        def __init__(self, data):
            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self):
        file="datasets/adult_cleaned.csv"
        trn, val, tst = adult_load_data_normalised(file)

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
      file="datasets/adult_cleaned.csv"
      trn, val, test = adult_load_data(file)
      data = np.vstack((trn, val))
      mu = data.mean(axis=0)
      s = data.std(axis=0)
      return mu, s

def adult_load_data(root_path):
    data = np.genfromtxt(root_path, delimiter=',', skip_header=1)
    rng = np.random.RandomState(42)
    rng.shuffle(data)
    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def adult_load_data_normalised(root_path):

    data_train, data_validate, data_test = adult_load_data(root_path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test

def get_adult_datasets(random_seed):
    print("Loading Adult dataset...")
    adult_dataset = AdultDataset()
    print("Adult dataset has been loaded!")
    return adult_dataset.train.x, adult_dataset.val.x, adult_dataset.test.x


def get_datasets(dataset_name, random_seed):
    """
    Returns train, val, test dataset
    """
    dataset_name = dataset_name.lower()
    mapping = {
        'mnist': get_mnist_datasets,
        'adult': get_adult_datasets
    #    'pums' : get_pums_datasets,
    #    'power' : get_power_datasets
     }
    if dataset_name not in mapping:
        err_msg = f"Unknown dataset '{dataset_name}'. Please choose one in {list(mapping.keys())}."
        raise ValueError(err_msg)

    return mapping[dataset_name](random_seed)

def get_input_size(dataset_name):
    """
    Returns the size of input
    """
    dataset_name = dataset_name.lower()
    mapping = {
        'mnist': 28 * 28,
        'adult': 6,
        'pums': 4,
        'power': 8
    }
    if dataset_name not in mapping:
        err_msg = f"Unknown dataset '{dataset_name}'. Please choose one in {list(mapping.keys())}."
        raise ValueError(err_msg)

    return mapping[dataset_name]

print('MADE & MAF Architecture')

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear

class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)

class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets

    def log_probs(self, inputs, cond_inputs = None):
        u, log_jacob = self(inputs, cond_inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode='inverse')[0]
        return samples

print('Training of the model')

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
            MADE(input_size, args.hidden_dims, num_cond_inputs=None, act='relu'),
            #fnn.BatchNormFlow(input_size),
            Reverse(input_size)
        ]
    model = FlowSequential(*modules)
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
        start = time.time()
        # Train for 1 epoch
        train_loss = 0
        if args.dataset_name == 'mnist':
          for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Loss = Negative Log Likelihood
            loss = -model.log_probs(batch).mean()
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        else:
          for batch in train_loader:
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
        if args.dataset_name == 'mnist':
          for batch, _ in val_loader:
            batch = batch.to(device)
            val_loss += -model.log_probs(batch).mean().item()
        else:
          for batch in val_loader:
            batch = batch.to(device)
            val_loss += -model.log_probs(batch).mean().item()
        avg_val_loss = np.sum(val_loss) / len(val_loader)

        end = time.time()
        duration = (end-start)/60

        if args.enable_dp:
          epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
        else:
          epsilon, best_alpha = None
        # Log statistics to wandb and stdout
        description = f'Epoch {epoch_num:3} | duration: {duration:12.5f}| train LL: {-avg_loss:12.5f} | val LL: {-avg_val_loss:12.5f} | epsilon: {epsilon:12.5f} | best alpha: {best_alpha:12.5f}'
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
    torch.save(model, "saved_models/" + args.dataset_name + "_trained_dp_model.pt")


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