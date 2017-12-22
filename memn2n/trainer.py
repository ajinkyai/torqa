import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .dataset import bAbIDataset
from .model import MemN2N


class Trainer():
    def __init__(self, config):
        self.train_data = bAbIDataset(config.dataset_dir, config.task)
        self.test_data = bAbIDataset(config.dataset_dir, config.task, train=False)
        pass

        settings = vars(config)
        settings['max_story_size'] = self.train_data.max_story_size
        settings['num_vocab'] = self.train_data.num_vocab
        settings['cuda'] = False
        memn2n = MemN2N(settings)
        memn2n.fit(self.train_data)
        acc = memn2n.tst(self.test_data)
        print('Final Test Accuracy: {}'.format(acc))