import os
from core.trainer import *

def get_args():
    device = "cuda"

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    trainer = Trainer()
    trainer.train()