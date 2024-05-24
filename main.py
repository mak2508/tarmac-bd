import os
import copy
import glob
import time

import torch
from arguments import get_args, setup_args

import pdb

if __name__ == "__main__":
    args = get_args()

    assert args.algo == 'a2c', "Only a2c is implemented for multiple agents"
    assert args.num_stack == 1, "Stacking frames not implemented for multiple agents"

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args = setup_args(args)

    if args.distributed == True and args.distributed_world_size > 1:
        print('Setting up distributed training')

        from train_dist import main
        main(args)
    else:
        print('Setting up single-gpu training')

        from train import main
        main(args)
