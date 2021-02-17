from datasets import get_datasets, get_input_size
from utils import get_distribution_diagnostic_plot, get_scattered_diagnostic_plot
from torch.utils.data import DataLoader
import flows

dataset = get_datasets("mnist", 42)
train, val, test = dataset
train_loader = DataLoader(train, batch_size=100)
val_loader = DataLoader(val, batch_size=100)
test_loader = DataLoader(test, batch_size=100)
data = next(iter(test_loader))[0]
get_distribution_diagnostic_plot("my_trained_maf.pt", data)
get_scattered_diagnostic_plot("my_trained_maf.pt", data)
