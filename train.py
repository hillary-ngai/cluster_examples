import torch
import argparse
import os
import time


def train(args):
    epoch = 0
    itr = 0
    model = torch.zeros(5)
    print('test')
    checkpoint_path = os.path.join(args.save_dir, 'checkpoint_last.pt')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        epoch = state_dict['epoch']
        itr = state_dict['itr']
        model = state_dict['model']
        args = state_dict['args']

    while epoch < args.max_epoch:
        print("========epoch {}========".format(epoch))
        while itr < args.max_itr:
            model += torch.ones(5)
            itr += 1
        epoch += 1
        itr = 0
        state_dict = {
            'args': args,
            'model': model,
            'epoch': epoch,
            'itr': itr
        }
        torch.save(state_dict, checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--max-itr', type=int, default=100)
    parser.add_argument('--save-dir', type=str, default="output")
    args = parser.parse_args()
    train(args)
