import os
from core.trainer import Trainer, get_args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer(args)
    trainer.train()