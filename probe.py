import copy
import glob
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from envs import make_env_multi_agent
from model import MultiAgentPolicy, RegressorProbe, ClassifierProbe
from storage import MultiAgentRolloutStorage

from gym.envs.registration import register

from tensorboardX import SummaryWriter
import shutil

import algo

import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
parser.add_argument(
    '--save-interval',
    type=int,
    default=10,
    help='save interval, one save per n updates (default: 10)')
parser.add_argument(
    '--vis-interval',
    type=int,
    default=100,
    help='vis interval, one log per n updates (default: 100)')
parser.add_argument(
    '--no-cuda',
    action='store_true',
    default=False,
    help='disables CUDA training')
parser.add_argument(
    '--port',
    type=int,
    default=8097,
    help='port to run the server on (default: 8097)')
parser.add_argument(
    '--no-vis',
    action='store_true',
    default=True,
    help='disables visdom visualization')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--num-processes',
    type=int,
    default=16,
    help='how many training CPU processes to use (default: 16)')
parser.add_argument(
    '--num-frames',
    type=int,
    default=1e9,
    help='number of frames to train (default: 10e6)')
parser.add_argument(
    '--recurrent-policy',
    action='store_true',
    default=True,
    help='use a recurrent policy')
parser.add_argument(
    '--save-dir',
    default='checkpoints/',
    help='directory to save agent logs (default: tmp/models)')

# Data/env parameters
parser.add_argument(
    '--data-dir',
    default='shapes_3x3_single_red',
    help='Set of images to use as env')
parser.add_argument('--task', default='colors.0', help='Single-target goal')
parser.add_argument(
    '--num-agents', type=int, default=4, help='number of agents')
parser.add_argument(
    '--step-size', type=int, default=2, help='size of a single step')
parser.add_argument(
    '--num-steps',
    type=int,
    default=30,
    help='number of forward steps in A2C (default: 5)')
parser.add_argument(
    '--obs-height', type=int, default=5, help='observation height')
parser.add_argument(
    '--obs-width', type=int, default=5, help='observation width')

# Multi-agent specific hyperparameters
parser.add_argument(
    '--env-name',
    default='ShapesEnv-v0',
    choices=['ShapesEnv-v0'],
    help='environment to train on (default: CartPole-v1)')
parser.add_argument(
    '--state-size',
    type=int,
    default=128,
    help='size of states in multi-agent setting')
parser.add_argument(
    '--comm-size', type=int, default=32, help='size of communication vectors')
parser.add_argument(
    '--comm-mode',
    default='from_states',
    choices=['from_states', 'no_comm', 'from_states_rec_att'])

# Logging
parser.add_argument('--log', action='store_true', default=False)
parser.add_argument(
    '--log-interval',
    type=int,
    default=20,
    help='log interval, one log per n updates (default: 20)')
parser.add_argument(
    '--tensorboard-dir',
    default='logs/tensorboard/',
    help='directory to save agent logs (default: tmp/tensorboard)')
parser.add_argument(
    '--log-dir',
    default='logs/',
    help='directory to save agent logs (default: tmp/gym)')

# Overfitting / spawn constraints
parser.add_argument('--overfit', action='store_true', default=False)
# Calling overfit fixes spawn locations and image; can also call them independently
parser.add_argument('--fix-spawn', action='store_true', default=False)
parser.add_argument('--fix-image', action='store_true', default=False)

# Probe parameters
parser.add_argument(
    '--probe-attr',
    default='all',
    choices=['coords', 'colors', 'shapes', 'all'])
parser.add_argument('--checkpoint-path', default=False)

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
args.vis = not args.no_vis

if args.overfit == True:
    args.fix_spawn = True
    args.fix_image = True


def main_multi_agent_probe():

    print(args.__dict__)

    # Re-register env to add custom args
    register(
        id='CustomShapesEnv-v0',
        entry_point='gym_shapes.shapes_env:ShapesEnv',
        kwargs={
            'data_dir': args.data_dir,
            'task': args.task,
            'num_agents': args.num_agents,
            'step_size': args.step_size,
            'num_steps': args.num_steps,
            'obs_height': args.obs_height,
            'obs_width': args.obs_width,
            'fix_spawn': args.fix_spawn,
            'fix_image': args.fix_image
        })

    # Parallel environments
    print('Setting up {} parallel envs'.format(args.num_processes))
    envs = [
        make_env_multi_agent('CustomShapesEnv-v0', args.seed, i)
        for i in range(args.num_processes)
    ]
    envs = SubprocVecEnv(envs) if (args.num_processes >
                                   1) else DummyVecEnv(envs)

    obs_shape = envs.observation_space.shape

    assert len(
        obs_shape) == 1, "CNN policy not implemented for multiple agents"
    assert envs.action_space.__class__.__name__ == "Discrete", "Continuous actions not implemented for multiple agents"

    # Load policy
    print('Loading model from {}'.format(args.checkpoint_path))
    if args.checkpoint_path == False:
        actor_critic = MultiAgentPolicy(args.num_agents, obs_shape[0],
                                        envs.action_space.n,
                                        args.recurrent_policy, args.state_size,
                                        args.comm_size, args.comm_mode)
    else:
        actor_critic = torch.load(
            args.checkpoint_path, map_location={
                'cuda:0': 'cpu'
            })

    actor_critic.eval()

    if args.cuda:
        actor_critic.cuda()

    print(actor_critic)

    # [TODO]
    # Load probes
    # Set up optimizers

    # Initial states
    rollouts = MultiAgentRolloutStorage(
        args.num_agents, obs_shape, args.num_steps, args.num_processes,
        actor_critic.state_size, actor_critic.comm_size)

    obs = envs.reset()
    rollouts.observations[0].copy_(torch.from_numpy(obs))

    if args.cuda:
        rollouts.cuda()

    num_updates = int(args.num_frames) // args.num_steps // args.num_processes

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            obs = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                value, actions, actions_log_probs, states, communications, aux = actor_critic.act(
                    rollouts.observations[step], rollouts.states[step],
                    rollouts.communications[step], rollouts.masks[step])

            # [TODO]
            # Need a way to query communication vectors here
            # Right now things get mean'ed / attended
            # .act() does return states, need a way to
            # just get intermediate outputs from the comm layers

            pdb.set_trace()


if __name__ == '__main__':

    main_multi_agent_probe()
