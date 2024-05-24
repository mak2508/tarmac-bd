import os
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

import gym
from gym import spaces

from House3D import objrender, Environment, load_config
from House3D.core import local_create_house

from gym_house3d.house3d import House3DUtils


class House3DEnv(gym.Env):
    def __init__(
            self,
            num_agents=4,
            num_steps=60,
            fix_env=False,
            fix_spawn=False,
            root_dir='/nethome/adas81/projects/embodied-qa/House3D',
            target_obj_conn_map_dir='/coc/scratch/abhshkdz/embodied-qa/target-obj-conn-maps/500'
    ):

        self.num_agents = num_agents
        self.num_steps = num_steps
        self.fix_env = fix_env
        self.fix_spawn = fix_spawn
        self.root_dir = root_dir
        self.target_obj_conn_map_dir = target_obj_conn_map_dir
        self.map_resolution = 500

        # Observation + OpenAI Gym parameters (for SubProcEnv)
        self.obs_height = 224
        self.obs_width = 224
        self.obs_channels = 3
        self.img_obs_shape = (self.obs_channels, self.obs_height, self.obs_width)
        self.action_space = spaces.Discrete(4) # forward, left, right, stay
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=self.img_obs_shape, dtype=np.float32)

        # [TODO]
        # maintain index of envs that satisfy goal requirements,
        # for example, if goal=find[fireplace], filter through envs
        # to keep only those containing fireplace.
        # Maybe precompute this list to have all sorts of filters
        # by object, by "big" objects, small objects, etc.

        # reserve renderer threads for envs to be loaded
        # supports only 1 gpu per training process for now
        # total of num_agents renderer threads
        self.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        self.cfg = load_config(
            os.path.join(self.root_dir, 'tests/config.json'))
        self.api_threads = []
        for i in range(self.num_agents):
            self.api_threads.append(objrender.RenderAPIThread(
                w=self.obs_width, h=self.obs_height, device=int(self.gpus[0])))
        self.env_loaded = False

        print('Loaded %d api threads on gpu %d' % (self.num_agents, int(self.gpus[0])))

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def reset(self, agent_positions=None, env_id=None):

        self.n_steps = 0

        # load houses
        if self.env_loaded == False:
            self.env_id = self._get_env_id()
            self.houses, self.envs = [], []
            for i in range(self.num_agents):
                self.houses.append(local_create_house(self.env_id, self.cfg, self.map_resolution))
                self.envs.append(House3DUtils(
                    Environment(self.api_threads[i], self.houses[-1], self.cfg),
                    target_obj_conn_map_dir=self.target_obj_conn_map_dir,
                    build_graph=False))
            self.env_loaded = True

            # set target object
            # hardcoded for "fireplace"
            # [TODO]
            # add support for more objects
            for i in range(self.num_agents):
                self.envs[i].set_target_object(
                    self.envs[i].objects['0_40'], '0_0')
            self.target_map = self.houses[-1].connMap

        agent_positions = self._get_agent_pos(agent_positions)
        self.agents = [Agent(*pos) for pos in agent_positions]

        joint_obs = self._get_joint_obs(reset=True)

        return joint_obs

    def step(self, actions):

        self.n_steps += 1

        joint_obs = np.zeros((self.num_agents, *self.img_obs_shape))
        reward = []

        for i in range(self.num_agents):
            obs, rwd, _ = self.envs[i].step(actions[i], step_reward=True)
            self.agents[i].set_pos(self.envs[i].env.cam.pos.x, self.envs[i].env.cam.pos.z, self.envs[i].env.cam.yaw)

            joint_obs[i] = self._preprocess_img_obs(obs)
            reward.append(rwd)

        # team reward
        # reward = np.sum(reward) / max(1, self.cars_in_sys)

        done = (self.n_steps == self.num_steps)

        return joint_obs, reward, done, {}

    def _get_env_id(self):
        if self.fix_env == True:
            return '75a7679d82f829167bc0a4f2136f7548'
        else:
            # [TODO]
            # add support for multiple envs
            raise KeyError

    def _get_agent_pos(self, agent_positions=None):
        if self.fix_spawn == True:
            spawn_loc_cands = np.argwhere(self.target_map == 50)

            pos = []
            for i in range(self.num_agents):
                # spawn_loc = spawn_loc_cands[np.random.choice(spawn_loc_cands.shape[0])]

                # fixed spawn location for all agents
                spawn_loc = spawn_loc_cands[0]

                cx, cy = self.houses[i].to_coor(spawn_loc[0], spawn_loc[1], shft=True)
                # pos.append([cx, cy, 18])
                # pos.append([cx, cy, np.random.choice([0, 9, 18, 27, 36])])
                pos.append([cx, cy, np.random.choice(self.envs[i].angles)])

            return pos

        # [TODO]
        # support more relaxed spawning
        # spawn_loc_cands = np.argwhere((self.target_map <= 50) & (self.target_map >= 20))
        spawn_loc_cands = np.argwhere((self.target_map == 30))

        pos = []
        for i in range(self.num_agents):
            # spawn_loc = spawn_loc_cands[np.random.choice(spawn_loc_cands.shape[0])]

            # fixed spawn location for all agents
            spawn_loc = spawn_loc_cands[np.random.choice(spawn_loc_cands.shape[0])]

            cx, cy = self.houses[i].to_coor(spawn_loc[0], spawn_loc[1], shft=True)
            # pos.append([cx, cy, 18])
            # pos.append([cx, cy, np.random.choice([0, 9, 18, 27, 36])])
            pos.append([cx, cy, np.random.choice(self.envs[i].angles)])

        return pos

    def _get_joint_obs(self, reset=False):
        # Image observations
        img_obs = np.zeros((self.num_agents, *self.img_obs_shape))

        for i, agent in enumerate(self.agents):
            img_obs[i] = self._get_img_obs(agent, self.envs[i], reset=reset)

        joint_obs = img_obs
        # joint_obs = img_obs.reshape(self.num_agents, -1)

        # Coordinate observations
        # if self.obs_coordinates:
        #     coordinates = np.zeros((self.num_agents, 2))

        #     for i, agent in enumerate(self.agents):
        #         coordinates[i] = np.array(agent.pos)

        #     if self.normalize:
        #         coordinates[:, 0] = coordinates[:, 0] / self.max_row
        #         coordinates[:, 1] = coordinates[:, 1] / self.max_col

        #     joint_obs = np.concatenate((joint_obs, coordinates), axis=1)

        # Time step observation
        # if self.obs_time_step:
        #     time_step = np.full((self.num_agents, 1), self.n_steps)

        #     if self.normalize:
        #         time_step = time_step / self.num_steps

        #     joint_obs = np.concatenate((joint_obs, time_step), axis=1)

        # # Agent ids
        # if self.obs_agent_ids:
        #     agents = np.zeros((self.num_agents, 1))
        #     for i, agent in enumerate(self.agents):
        #         agents[i] = i / len(self.agents)
        #     joint_obs = np.concatenate((joint_obs, agents), axis=1)

        # # Goal observations
        # if self.obs_goals:
        #     goals = np.zeros((self.num_agents, 1))

        #     for i, agent in enumerate(self.agents):
        #         goals[i] = self.task_id[i]

        #     joint_obs = np.concatenate((joint_obs, goals), axis=1)

        return joint_obs

    def _get_img_obs(self, agent, env, reset=False, preprocess=True):
        if reset == True:
            env.env.reset(x=agent.x, y=agent.y, yaw=agent.yaw)

        obs = np.array(env.env.render(), copy=False, dtype=np.float32)

        if preprocess == True:
            obs = self._preprocess_img_obs(obs)

        return obs

    def _preprocess_img_obs(self, obs):
        obs = obs.transpose(2, 0, 1)
        obs = obs / 255.0

        return obs

    def render(self, save_dir="", aux={}):

        hdim = 2
        if 'p_attn' in aux:
            hdim += 1

        fig = plt.figure(figsize=(10 * hdim, 10))
        outer = gridspec.GridSpec(1, hdim, wspace=0.2)

        inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[0])
        ax = plt.Subplot(fig, inner[0])

        viz_map = self.target_map.copy()

        viz_map.fill(0)
        viz_map[(self.target_map >= 0) & (self.target_map <= 20)] = 1000
        viz_map[self.target_map > 0] = 500

        ax.imshow(viz_map[:, ::-1])
        for i, _ in enumerate(self.agents):
            gx, gy = self.envs[i].env.house.to_grid(self.agents[i].x, self.agents[i].y)
            # plot agent location
            ax.plot(self.target_map.shape[0] - gy, gx, '*', color='red')
            # plot agent yaw
            ax.plot([self.target_map.shape[0] - gy, self.target_map.shape[0] - gy - 10 * np.sin(self.agents[i].yaw * np.pi / 180)],
                    [gx, gx + 10 * np.cos(self.agents[i].yaw * np.pi / 180)], '-', color='white')
            # plot agent id
            ax.text(self.target_map.shape[0] - gy - 5, gx - 5, str(i+1), fontsize=10, color='white')

        ax.tick_params(
            which='both',
            left=False,
            bottom=False,
            labelbottom=False,
            labelleft=False)
        fig.add_subplot(ax)

        inner = gridspec.GridSpecFromSubplotSpec(self.num_agents//2, 2,
                    subplot_spec=outer[1])
        for i, _ in enumerate(self.agents):
            ax = plt.Subplot(fig, inner[i])
            ax.imshow(self._get_img_obs(self.agents[i], self.envs[i], preprocess=False) / 255.0)
            ax.tick_params(
                which='both',
                left=False,
                bottom=False,
                labelbottom=False,
                labelleft=False)
            ax.set_xlabel('Agent %d' % (i+1), fontsize=20)
            fig.add_subplot(ax)

        if 'p_attn' in aux:
            inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                        subplot_spec=outer[2])
            ax = plt.Subplot(fig, inner[0])
            aux['p_attn'] = aux['p_attn'][-1].data.numpy()
            ax.imshow(aux['p_attn'], cmap='Blues', vmin=0.0, vmax=1.0)
            ax.tick_params(
                which='both',
                left=False,
                bottom=False,
                labelbottom=False,
                labelleft=False)
            fig.add_subplot(ax)

        if save_dir is not "":
            plt.savefig("%s/%05d" % (save_dir, self.n_steps))
        fig.show()

class Agent:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

    @property
    def pos(self):
        return self.x, self.y, self.yaw

    def set_pos(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw

if __name__ == '__main__':
    env = House3DEnv()