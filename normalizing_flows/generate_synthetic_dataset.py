import argparse

import numpy as np
import torch
from dataset_loader import get_input_size
import flows as fnn

def main(args):
    n_dims = get_input_size(args.dataset_name)
    inputs = torch.Tensor(args.n_samples, n_dims).normal_().to('cuda')
    model = torch.load(args.model_path)
    model.to("cuda")
    with torch.no_grad():
        for module in reversed(model._modules.values()):
            if isinstance(module, fnn.Reverse):
                inputs = inputs[:, np.argsort(module.perm)]
            elif isinstance(module, fnn.MADE):
                tmp = torch.zeros_like(inputs)
                for i_col in range(inputs.shape[1]):
                    h = module.joiner(tmp)
                    m, a = module.trunk(h).chunk(2, 1)
                    tmp[:, i_col] = inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
                inputs = tmp
            elif isinstance(module, fnn.BatchNormFlow):
                #mean = module.batch_mean
                #var = module.batch_var
                mean = module.running_mean
                var = module.running_var
                x_hat = (inputs - module.beta) / torch.exp(module.log_gamma)
                y = x_hat * var.sqrt() + mean
                inputs = y
            else:
                raise ValueError("Unknown module type in the flow: {0}".format(type(module)))
    synth_data_df = inputs.detach().cpu().numpy()
    print(synth_data_df)
    pd.DataFrame(synth_data_df).to_csv('synth_data/synth_'+args.dataset_name + '.csv')
      
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate synthetic dataset")
    parser.add_argument('--use_cuda', default=True, type=bool, help="Whether to use GPU or CPU. True for GPU")
    parser.add_argument('--dataset_name', default='mnist', type=str, help="Dataset name to train on")
    parser.add_argument('--n_samples', default=10, type=int, help="Number of rows of synthetic data to be generated")
    parser.add_argument('--model_path', default='./saved_models/mnist_trained_model.pt', type=str, help='File path to the saved model')
    args = parser.parse_args()
    main(args)
