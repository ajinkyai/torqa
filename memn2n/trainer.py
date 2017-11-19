import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .dataset import bAbIDataset


class Trainer():
    def __init__(self, config):
        self.train_data = bAbIDataset(config.dataset_dir, config.task)
        self.test_data = bAbIDataset(config.dataset_dir, config.task, train=False)
        pass