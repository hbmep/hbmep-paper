import os
import random
import logging

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpyro.distributions as dist

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hbmep.config import Config
from hbmep.model import BaseModel
from hbmep.model import functional as F

from hbmep_paper.utils import setup_logging

logger = logging.getLogger(__name__)

TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/experiments/tms.toml"
BUILD_DIR = "/home/vishu/repos/hbmep-paper/reports/experiments/tms/auto-differentiation-opt"

# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Set device
cuda_status = torch.cuda.is_available()
device = torch.device("cuda" if cuda_status else "cpu")


class CustomDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match the length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y


def make_data_loader(X, Y, batch_size=32):
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    data_loader = DataLoader(CustomDataset(X, Y), batch_size=32, shuffle=True)
    return data_loader


def main():
    toml_path = TOML_PATH
    config = Config(toml_path=toml_path)
    config.BUILD_DIR = BUILD_DIR
    model = BaseModel(config=config)

    # Set up logging
    model._make_dir(model.build_dir)
    setup_logging(dir=model.build_dir, fname=os.path.basename(__file__))

    # y = b(x - a)
    b_true = 2.0
    a_true = 1.0

    # Simulate data
    num_points = 50
    x = np.linspace(0, 10, 50)
    y_true = b_true * (x - a_true)

    # Add Gaussian noise
    y_noise = dist.Normal(0, 1).sample(model.rng_key, (num_points,))
    y_noise = np.array(y_noise)
    logger.info(y_noise.shape)
    y_obs = y_true + y_noise

    # Plot simulated data
    fig, axes = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, constrained_layout=True)
    ax = axes[0, 0]
    sns.scatterplot(x=x, y=y_obs, ax=ax, label="Observed")
    sns.lineplot(x=x, y=y_true, ax=ax, color="black", label="True")
    dest = os.path.join(model.build_dir, "simulated_data.png")
    fig.savefig(dest)
    logger.info(f"Saved to {dest}")

    # Regression using autograd
    a = torch.randn((1,), requires_grad=True)
    b = torch.randn((1,), requires_grad=True)
    logger.info(f"a: {a}, {a.shape}")
    logger.info(f"b: {b}, {b.shape}")
    logger.info(f"device: {a.get_device()}")

    data_loader = make_data_loader(x, y_obs)
    # y = b * (x - a)
    # loss = 
    return


if __name__ == "__main__":
    main()