import torch
from torchvision import transforms

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random

class StockDataset(Dataset):
    def __init__(self):
        self.trade_train = torch.tensor(pd.read_csv('data/trade_train.csv'))
        self.stocks = torch.tensor(pd.read_csv('data/stocks.csv'))
        self.length = len(self.trade_train)

    def __len__(self):
        return self.length

    def __getitem__(self, i):        
        return self.trade_train[i]