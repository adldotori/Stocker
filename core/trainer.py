import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import get_args
from .model import Stock
from .dataloader import StockLoader

class Trainer():
    def __init__(self, args):
        self.args = args

        self.loader = StockLoader(args)
        self.model = Stock(args)
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters())

    def train(self):
        for epoch in range(self.args.epoch):
            for i in range(len(self.loader.data_loader)):
                group_info, month_info, day_info, label = self.loader.next_batch()
                
                pred = self.model(group_info, month_info, day_info)

                loss = self.loss(pred, label)
                loss.backward()
                self.optim.step()
            print(f'{epoch}:{loss}')
            
            torch.save(self.model.state_dict(), 'checkpoint.pt')
