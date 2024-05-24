import os
import copy
import glob
import time

import argparse
import torch
import torch.distributed

# Traffic-env constants
traffic_env_config = {
    'easy': {
        'dim': 6,
        'vision': 1,
        'num_agents': 4,
        'difficulty': 'easy',
        'add_rate': 0.30
    },
    'medium': {
        'dim': 14,
        'vision': 1,
        'num_agents': 10,
        'difficulty': 'medium',
        'add_rate': 0.20
    },
    'hard': {
        'dim': 18,
        'vision': 1,
        'num_agents': 20,
        'difficulty': 'hard',
        'add_rate': 0.05
    },
    'gcc-easy': {
        'dim': 6,
        'vision': 0,
        'num_agents': 4,
        'difficulty': 'easy'
    },
    'gcc-medium': {
        'dim': 14,
        'vision': 0,
        'num_agents': 10,
        'difficulty': 'medium'
    },
    'gcc-hard': {
        'dim': 18,
        'vision': 0,
        'num_agents': 20,
        'difficulty': 'hard'
    }
}


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.95,
        help='gae parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--num-stack',
        type=int,
        default=1,
        help='number of frames to stack (default: 1)')
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
        '--env-name',
        default='ShapesEnv-v0',
        choices=['ShapesEnv-v0', 'TrafficEnv-v0', 'House3DEnv-v0'],
        help='environment to train on (default: ShapesEnv-v0)')
    # SHAPES-specific
    parser.add_argument(
        '--num-agents', type=int, default=4, help='number of agents')
    parser.add_argument(
        '--data-dir',
        default='shapes_3x3_single_red',
        help='Set of images to use as env')
    parser.add_argument(
        '--task', default='colors.4,0,0', help='Single-target goal')
    parser.add_argument(
        '--obs-height', type=int, default=5, help='observation height')
    parser.add_argument(
        '--obs-width', type=int, default=5, help='observation width')
    parser.add_argument(
        '--step-size', type=int, default=2, help='size of a single step')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=30,
        help='number of forward steps in A2C (default: 5)')
    # Traffic-specific
    parser.add_argument(
        '--difficulty',
        default='easy',
        help='Difficulty setting for the traffic env',
        choices=['easy', 'medium', 'hard', 'gcc-easy', 'gcc-medium', 'gcc-hard'])
    parser.add_argument(
        '--traffic-num-steps',
        type=int,
        default=60,
        help='number of forward steps in A2C (default: 5)')
    # House3D-specific
    parser.add_argument('--fix-env', action='store_true', default=False)
    parser.add_argument('--use-cnn', action='store_true', default=False)

    # Multi-agent specific hyperparameters
    parser.add_argument(
        '--state-size',
        type=int,
        default=128,
        help='size of states in multi-agent setting')
    parser.add_argument(
        '--comm-size',
        type=int,
        default=32,
        help='size of communication vectors')
    parser.add_argument(
        '--comm-num-hops',
        type=int,
        default=1,
        help='no. of communication hops per action')
    parser.add_argument(
        '--comm-mode',
        default='from_states',
        choices=['from_states', 'no_comm', 'from_states_rec_att'])

    # Distributed training
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument(
        '--distributed-world-size',
        type=int,
        metavar='N',
        default=torch.cuda.device_count(),
        help='total number of GPUs across all nodes (default: all visible GPUs)'
    )
    parser.add_argument(
        '--distributed-rank',
        default=0,
        type=int,
        help='rank of the current worker')
    parser.add_argument(
        '--distributed-backend',
        default='tcp',
        type=str,
        help='distributed backend')
    parser.add_argument(
        '--distributed-init-method',
        default=None,
        type=str,
        help='typically tcp://hostname:port that will be used to '
        'establish initial connection')
    parser.add_argument(
        '--distributed-port',
        default=-1,
        type=int,
        help='port number (not required if using --distributed-init-method)')
    parser.add_argument(
        '--device-id',
        default=0,
        type=int,
        help='which GPU to use (usually configured automatically)')

    # Logging
    parser.add_argument('--identifier', default=None)
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

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.vis = not args.no_vis

    if args.overfit == True:
        args.fix_spawn = True
        args.fix_image = args.fix_env = True

    return args


def setup_args(args):

    args.num_updates = int(
        args.num_frames) // args.num_steps // args.num_processes

    if args.env_name == 'House3DEnv-v0':
        args.use_cnn = True

    if args.log == True:
        time_id = time.strftime("%m_%d")
        args.tensorboard_dir = '%s/%s/tensorboard/' % (args.log_dir, time_id)
        args.save_dir = '%s/%s/' % (args.save_dir, time_id)
        args.log_dir = '%s/%s/' % (args.log_dir, time_id)

        hparam_str = "{}_{}".format(
            time.strftime('%m_%d_%H%M%S'), args.env_name)

        if args.identifier != None:
            hparam_str += ',{}'.format(args.identifier)

        if args.env_name == 'ShapesEnv-v0':
            hparam_str += ",data={},task={},num_agents={},comm_mode={},comm_nhops={},step_size={},num_steps={},state_size={},comm_size={},fix_spawn={},fix_image={}".format(
                args.data_dir, args.task, str(args.num_agents), str(args.comm_mode),
                str(args.comm_num_hops), str(args.step_size), str(args.num_steps),
                str(args.state_size), str(args.comm_size), str(args.fix_spawn),
                str(args.fix_image))
        elif args.env_name == 'TrafficEnv-v0':
            hparam_str += ",difficulty={},comm_mode={},comm_nhops={},state_size={},comm_size={}".format(
                args.difficulty, str(args.comm_mode), str(args.comm_num_hops),
                str(args.state_size), str(args.comm_size))
        elif args.env_name == 'House3DEnv-v0':
            hparam_str += ",num_agents={},num_steps={},comm_mode={},comm_nhops={},state_size={},comm_size={}".format(
                str(args.num_agents), str(args.num_steps),
                str(args.comm_mode), str(args.comm_num_hops),
                str(args.state_size), str(args.comm_size))

        args.log_dir = args.log_dir + hparam_str + "/"
        args.save_dir = args.save_dir + hparam_str + "/"
        args.tensorboard_dir = args.tensorboard_dir + hparam_str + "/"

        # Check log directories exist and are empty
        try:
            os.makedirs(args.tensorboard_dir)
        except OSError:
            shutil.rmtree(args.tensorboard_dir)
            os.makedirs(args.tensorboard_dir)

        try:
            os.makedirs(args.save_dir)
        except OSError:
            shutil.rmtree(args.save_dir)
            os.makedirs(args.save_dir)

        try:
            os.makedirs(args.log_dir)
        except OSError:
            files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

    return args
