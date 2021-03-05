import argparse

from dataset_loader import get_datasets, get_input_size
from utils import get_distribution_diagnostic_plot, get_scattered_diagnostic_plot
from torch.utils.data import DataLoader
import flows

def main(args): 
  dataset = get_datasets(args.data_name, 42)
  train, val, test = dataset
  train_loader = DataLoader(train, batch_size=100)
  val_loader = DataLoader(val, batch_size=100)
  test_loader = DataLoader(test, batch_size=100)
  if args.data_name == "mnist":
    data = next(iter(test_loader))[0]
  else:
    data = next(iter(test_loader))
  # print('get_distribution_plot', args.model_path, args.data_name)
  get_distribution_diagnostic_plot(args.model_path, data, args.data_name)
  # print('get_scattered_diagnostic_plot')
  get_scattered_diagnostic_plot(args.model_path, data, args.data_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script to train differentially private normalizing flows model")
    parser.add_argument('--data_name', default='mnist', type=str, help="Name of dataset used in model")
    parser.add_argument('--model_path', default="./saved_models/mnist_trained_model.pt", type=str, help="Path to trained model")
    args = parser.parse_args()
    main(args)