from gccnet_envs.traffic_junction_env import TrafficJunctionEnv

import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle

from gym import spaces


class TrafficEnv(TrafficJunctionEnv):
    def __init__(self,
                 num_agents=4,
                 dim=18,
                 vision=1,
                 add_rate_min=0.05,
                 add_rate_max=0.20,
                 difficulty='hard',
                 num_steps=60):

        TrafficJunctionEnv.__init__(self)

        self.num_agents = num_agents
        self.num_steps = num_steps

        args = {
            'nagents': num_agents,
            'dim': dim,
            'vision': vision,
            'add_rate_min': add_rate_min,
            'add_rate_max': add_rate_max,
            'curr_start': 0,
            'curr_end': 0,
            'difficulty': difficulty,
            'vocab_type': 'bool'
        }

        args = namedtuple("args", args.keys())(*args.values())

        self.multi_agent_init(args)

        if self.vocab_type == 'bool':
            # actions
            obs_len = 1

            # paths
            obs_len += 1

            # vision
            obs_len += (2 * vision + 1) * (2 * vision + 1) * self.vocab_size

            self.observation_space = spaces.Box(
                low=0.0, high=1.0, shape=(obs_len, ), dtype=np.float32)
        else:
            raise KeyError

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, ):

        self.n_steps = 0

        obs = TrafficJunctionEnv.reset(self)

        joint_obs = np.zeros((self.num_agents,
                              self.observation_space.shape[0]))
        for i in range(self.num_agents):
            joint_obs[i] = np.concatenate((np.array((obs[i][0], obs[i][1])),
                                           obs[i][2].flatten()))

        return joint_obs

    def step(self, actions):

        self.n_steps += 1

        obs, reward, done, info = TrafficJunctionEnv.step(self, actions)

        joint_obs = np.zeros((self.num_agents,
                              self.observation_space.shape[0]))
        for i in range(self.num_agents):
            joint_obs[i] = np.concatenate((np.array((obs[i][0], obs[i][1])),
                                           obs[i][2].flatten()))

        # team reward
        # reward = np.sum(reward) / max(1, self.cars_in_sys)

        done = (self.n_steps == self.num_steps)

        info['success'] = self.stat['success']
        info['add_rate'] = self.stat['add_rate']

        return joint_obs, reward, done, info

    def render(self, save_dir="", aux={}):
        cell_sz = 5
        car_grid = np.ones(((self.dim + 1) * cell_sz, (self.dim + 1) * cell_sz,
                            3))

        # render each car + check for collisions
        # blue for cars, red for collisions
        car_loc_dict = {}
        for i in range(self.num_agents):
            if self.alive_mask[i] == 0:
                continue
            car_loc_dict[(self.car_loc[i][0], self.car_loc[i][1])] = car_loc_dict.get((self.car_loc[i][0], self.car_loc[i][1]), 0) + 1
            if car_loc_dict[(self.car_loc[i][0], self.car_loc[i][1])] > 1:
                car_grid[cell_sz * self.car_loc[i][0]:cell_sz * (
                    self.car_loc[i][0] + 1), cell_sz * self.car_loc[i][1]:cell_sz *
                         (self.car_loc[i][1] + 1), 1:] = 0.0
                car_grid[cell_sz * self.car_loc[i][0]:cell_sz * (
                    self.car_loc[i][0] + 1), cell_sz * self.car_loc[i][1]:cell_sz *
                         (self.car_loc[i][1] + 1), 0] = 1.0
            else:
                car_grid[cell_sz * self.car_loc[i][0]:cell_sz * (
                    self.car_loc[i][0] + 1), cell_sz * self.car_loc[i][1]:cell_sz *
                         (self.car_loc[i][1] + 1), :2] = 0.0

        hdim = 1
        if 'p_attn' in aux:
            hdim += 1

        fig = plt.figure(figsize=(8 * hdim, 8))

        ax = plt.subplot(1, hdim, 1)
        plt.imshow(car_grid)

        for i in range(self.num_agents):
            if self.alive_mask[i] == 0:
                continue
            ax.text(
                cell_sz * self.car_loc[i][1],
                cell_sz * self.car_loc[i][0] + cell_sz / 2,
                str(i + 1),
                color='white')

        ticks = ticker.FuncFormatter(
            lambda x, pos: '{0:g}'.format(x / cell_sz))
        loc = ticker.MultipleLocator(base=cell_sz)
        ax.xaxis.set_major_formatter(ticks)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_formatter(ticks)
        ax.yaxis.set_major_locator(loc)

        plt.title('Traffic Junction')

        if 'p_attn' in aux:
            ax = plt.subplot(1, hdim, 2)

            attn = aux['p_attn'][0].data.numpy()
            for i in range(1, len(aux['p_attn'])):
                attn += aux['p_attn'][i].data.numpy()
            attn /= len(aux['p_attn'])

            # aux['p_attn'] = aux['p_attn'][-1].data.numpy()
            plt.imshow(attn, cmap='Blues', vmin=0.0, vmax=1.0)

            plt.xlabel('Sender')
            plt.ylabel('Receiver')

            ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x + 1))
            loc = ticker.MultipleLocator(base=1.0)
            ax.xaxis.set_major_formatter(ticks)
            ax.xaxis.set_major_locator(loc)
            ax.yaxis.set_major_formatter(ticks)
            ax.yaxis.set_major_locator(loc)

            plt.title('Communication attention')

        if save_dir is not "":
            plt.savefig("%s/%05d" % (save_dir, self.n_steps))


if __name__ == '__main__':
    env = TrafficEnv()