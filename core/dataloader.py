import torch
from torchvision import transforms
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import random


class StockDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.trade = pd.read_csv('data/trade_train.csv')
        self.stocks = pd.read_csv('data/stocks.csv')
        self.pidak = pd.read_csv('data/KospiKosdaq.csv')
        # self.large_class = pd.read_csv('data/large_class.csv')
        # self.medium_class = pd.read_csv('data/medium_class.csv')
        # self.small_class = pd.read_csv('data/small_class.csv')



    def __len__(self):
        return len(self.pidak)

    def __getitem__(self, i):        
        return torch.randn(4),\
               torch.randn(12, 2),\
               torch.randn(12, 25, 2)

class StockLoader(object):
    def __init__(self, args, dataset):
        super().__init__()
        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':
    from args import get_args 
    args = get_args()

    dataset = StockDataset(args)
    loader = StockLoader(args, dataset)

    a, b, c = loader.next_batch()
    print(a.shape, b.shape, c.shape)