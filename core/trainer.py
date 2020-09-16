import torch
import torch.nn as nn
import torch.nn.functional as F

from .args import get_args
from .model import Stock
from .dataloader import StockLoader

class Trainer():
    def __init__(self, args):
        self.args = args

        self.dataloader = StockLoader(args)
        self.model = Stock(args)
    def train(self):
        for i in range(self.args.epoch):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                torch.zeros(1, 1, model.hidden_layer_size))

                y_pred = model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i%25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')