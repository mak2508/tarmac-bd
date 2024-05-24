import os
import sys
import json
import random
import argparse

import pdb
import torch

from train import main as single_process_main


def main(args):
    torch.distributed.init_process_group(
        backend='tcp',
        init_method=args.distributed_init_method,
        world_size=args.distributed_world_size,
        rank=args.distributed_rank)

    single_process_main(args)


# def run(args):

#     torch.distributed.init_process_group(
#         backend='tcp',
#         init_method='tcp://localhost:9218',
#         world_size=args.distributed_world_size,
#         rank=args.distributed_rank)

#     args.distributed_rank = torch.distributed.get_rank()
#     single_process_main(args)
