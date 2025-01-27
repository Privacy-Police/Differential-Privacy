import torch
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from datasets import adult
from datasets import pums
from datasets import power


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

def get_adult_datasets(random_seed):
    print("Loading Adult dataset...")
    adult_dataset = adult.AdultDataset()
    print("Adult dataset has been loaded!")
    return adult_dataset.train.x, adult_dataset.val.x, adult_dataset.test.x
    
def get_pums_datasets(random_seed):
    print("Loading PUMS dataset...")
    pums_dataset = pums.PUMSDataset()
    print("PUMS dataset has been loaded!")
    return pums_dataset.train.x, pums_dataset.val.x, pums_dataset.test.x
    
def get_power_datasets(random_seed):
    print("Loading POWER dataset...")
    power_dataset = power.PowerDataset()
    print("POWER dataset has been loaded!")
    return power_dataset.train.x, power_dataset.val.x, power_dataset.test.x

def get_datasets(dataset_name, random_seed):
    """
    Returns train, val, test dataset
    """
    dataset_name = dataset_name.lower()
    mapping = {
       'mnist': get_mnist_datasets,
       'adult' : get_adult_datasets,
       'pums' : get_pums_datasets,
       'power' : get_power_datasets
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

def get_adult_stats():
    return adult.AdultDataset().get_stats()
    
def get_pums_stats():
    return pums.PUMSDataset().get_stats()
    
def get_power_stats():
    return power.PowerDataset().get_stats()
  
def get_dataset_stats(dataset_name):
    """
    Returns the mean, var of dataset
    """
    dataset_name = dataset_name.lower()
    mapping = {
        'adult': get_adult_stats,
        'pums': get_pums_stats,
        'power': get_power_stats
    }
    if dataset_name not in mapping:
        err_msg = f"Unknown dataset '{dataset_name}'. Please choose one in {list(mapping.keys())}."
        raise ValueError(err_msg)

    return mapping[dataset_name]()