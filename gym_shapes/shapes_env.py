import os
import json
import copy
import random
import itertools

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import numpy as np

import gym
from gym import spaces


class ShapesEnv(gym.Env):
    def __init__(
            self,
            data_root_dir='gym_shapes/data/shapes/data/',
            data_dir='shapes_3x3_single_red',
            split='train.large',  # train.med, train.small, train.tiny, val, test
            # {colors, shapes, sizes} x {0, 1, 2}
            # colors.4,0,0 means
            # 4 agents find red, 0 find green, 0 find blue
            task='colors.4,0,0',
            num_agents=4,
            step_size=1,
            num_steps=30,
            obs_height=5,
            obs_width=5,
            fix_spawn=False,
            fix_image=False,
            obs_coordinates=True,
            obs_time_step=True,
            obs_goals=True,
            obs_agent_ids=False,
            normalize=True):

        self.num_agents = num_agents
        self.step_size = step_size
        self.num_steps = num_steps
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.fix_spawn = fix_spawn
        self.fix_image = fix_image
        self.obs_coordinates = obs_coordinates
        self.obs_time_step = obs_time_step
        self.obs_goals = obs_goals
        self.obs_agent_ids = obs_agent_ids
        self.normalize = normalize

        # Load data
        self.data_root = os.path.join(data_root_dir, data_dir, split)

        self.data = np.load(self.data_root + '.input.npy')[:, :, :, ::-1]
        self.attr = {
            'shapes': json.load(open(self.data_root + '.shapes', 'r')),
            'colors': json.load(open(self.data_root + '.colors', 'r')),
            'sizes': json.load(open(self.data_root + '.sizes', 'r'))
        }

        self.n_imgs = self.data.shape[0]
        self.img_shape = self.data[0].shape
        self.img_obs_shape = (self.obs_height, self.obs_width,
                              self.img_shape[2])

        self._parse_task(task)

        self.max_row = self.img_shape[0] - self.obs_height
        self.max_col = self.img_shape[1] - self.obs_width

        # [NOTE] these constants assume 30 x 30 image has 3x3 grid
        self.max_row_cells = int(self.img_shape[0] / 10)
        self.max_col_cells = int(self.img_shape[1] / 10)

        # Gym parameters
        self._action_space = spaces.Discrete(5)

        len_obs = self.img_obs_shape[0] * self.img_obs_shape[
            1] * self.img_obs_shape[2]
        if self.obs_coordinates:
            len_obs += 2
        if self.obs_time_step:
            len_obs += 1
        if self.obs_agent_ids:
            len_obs += 1
        if self.obs_goals:
            len_obs += 1
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len_obs, ), dtype=np.float32)

    def reset(self, agent_positions=None, img_idx=None):
        self.n_steps = 0

        self.visited_states = []
        self.max_states = (self.max_row // self.step_size) * (
            self.max_col // self.step_size)

        self.image_idx = idx = self._get_image_id(img_idx)
        attr = self.attr[self.task][idx]

        # make sure sampled image has all goals
        while len([
                x for x in list(set(itertools.chain.from_iterable(attr)))
                if isinstance(x, int)
        ]) != 3:
            idx = np.random.randint(self.n_imgs)
            attr = self.attr[self.task][idx]

        self.image = copy.deepcopy(self.data[idx])
        self.success_maps = [np.array(attr) == i for i in range(3)]

        agent_positions = self._get_agent_pos(agent_positions)
        self.agents = [Agent(*pos) for pos in agent_positions]

        joint_obs = self._get_joint_obs()

        return joint_obs

    def step(self, joint_action):
        info = {}

        self.n_steps += 1

        for agent, action in zip(self.agents, joint_action):
            self._move(agent, action)

        joint_obs = self._get_joint_obs()
        # team reward
        # reward = sum([
        #     self._on_goal(self.agents[i], self.success_maps[self.task_id[i]])
        #     for i in range(self.num_agents)
        # ]) / self.num_agents
        # individual reward
        reward = np.array([
            self._on_goal(self.agents[i], self.success_maps[self.task_id[i]])
            for i in range(self.num_agents)
        ], dtype=np.float32)
        done = (self.n_steps == self.num_steps)

        info['success'] = self.is_success()
        info['coverage'] = self.get_coverage()
        info['image_idx'] = self.image_idx
        info['n_steps'] = self.n_steps

        return joint_obs, reward, done, info

    def render(self, save_dir="", aux={}):
        visions = np.zeros(self.img_shape, dtype=int)
        positions = np.ones(self.img_shape, dtype=int) * 255

        for agent in self.agents:
            r, c = agent.pos
            h, w = self.obs_height, self.obs_width
            visions[r:r + h, c:c + w, :] = self.image[r:r + h, c:c + w, :]

        hdim = 3
        if 'p_attn' in aux:
            hdim += 1

        fig = plt.figure(figsize=(4 * hdim, 4))

        ax = plt.subplot(1, hdim, 1)
        plt.imshow(self.image)
        for i in range(len(self.agents)):
            r, c = self.agents[i].pos
            p = Rectangle(
                (c - 0.5, r - 0.5),
                self.obs_width,
                self.obs_height,
                fill=False,
                edgecolor='white')
            ax.add_patch(p)
            ax.text(c + 1.5, r + 3, str(i + 1), color='white')
        plt.title('Complete state')

        ax = plt.subplot(1, hdim, 2)
        plt.imshow(visions)
        for i in range(len(self.agents)):
            r, c = self.agents[i].pos
            p = Rectangle(
                (c - 0.5, r - 0.5),
                self.obs_width,
                self.obs_height,
                fill=False,
                edgecolor='white')
            ax.add_patch(p)
            ax.text(c + 1.5, r + 3, str(i + 1), color='white')
        plt.title('Observed state (jointly by all)')

        ax = plt.subplot(1, hdim, 3)
        for i in range(len(self.agents)):
            r, c = self.agents[i].pos
            plt_r = 14 * int(i // (self.max_col / (self.obs_width + 2)))
            plt_c = 2 + 7 * int(i % (self.max_col / (self.obs_width + 2)))
            positions[plt_r:plt_r + self.obs_height, plt_c:
                      plt_c + self.obs_width] = self.image[r:r + h, c:c + w, :]
            ax.text(
                plt_c,
                plt_r + 7,
                "(%02d,%02d)" % (r, c),
                color='black',
                fontsize=6)
            ax.text(plt_c + 1.5, plt_r + 10, str(i + 1), color='black')
        ax.axis('off')
        plt.imshow(positions)
        plt.title('Agent views & positions')
        plt.tick_params(
            which='both',
            left=False,
            bottom=False,
            labelbottom=False,
            labelleft=False)

        if 'p_attn' in aux:
            ax = plt.subplot(1, hdim, 4)
            aux['p_attn'] = aux['p_attn'].data.numpy()
            plt.imshow(aux['p_attn'], cmap='Blues', vmin=0.0, vmax=1.0)

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
        plt.show()

    def _get_image_id(self, img_idx=None):
        if self.fix_image == True:
            return 25

        if img_idx != None:
            return img_idx
        else:
            return np.random.randint(self.n_imgs)

    def _get_agent_pos(self, agent_positions=None):
        if self.fix_spawn == True:
            # corners
            pos = []
            for i in range(self.num_agents):
                if i % 4 == 0:
                    pos.append((0, 0))
                elif i % 4 == 1:
                    pos.append((0, self.max_col))
                elif i % 4 == 2:
                    pos.append((self.max_row, self.max_col))
                elif i % 4 == 3:
                    pos.append((self.max_row, 0))
            return pos

        if agent_positions != None:
            return agent_positions
        else:
            return [(random.randint(0, self.max_row),
                     random.randint(0, self.max_col))
                    for _ in range(self.num_agents)]

    def _get_joint_obs(self):
        # Image observations
        img_obs = np.zeros((self.num_agents, *self.img_obs_shape))

        for i, agent in enumerate(self.agents):
            img_obs[i] = self._get_img_obs(agent)

        joint_obs = img_obs.reshape(self.num_agents, -1)

        # Coordinate observations
        if self.obs_coordinates:
            coordinates = np.zeros((self.num_agents, 2))

            for i, agent in enumerate(self.agents):
                coordinates[i] = np.array(agent.pos)

            if self.normalize:
                coordinates[:, 0] = coordinates[:, 0] / self.max_row
                coordinates[:, 1] = coordinates[:, 1] / self.max_col

            joint_obs = np.concatenate((joint_obs, coordinates), axis=1)

        # Time step observation
        if self.obs_time_step:
            time_step = np.full((self.num_agents, 1), self.n_steps)

            if self.normalize:
                time_step = time_step / self.num_steps

            joint_obs = np.concatenate((joint_obs, time_step), axis=1)

        # Agent ids
        #
        # Sequence doesn't mean much
        # should be used to index a lookup table
        if self.obs_agent_ids:
            agents = np.zeros((self.num_agents, 1))
            for i, agent in enumerate(self.agents):
                agents[i] = i / len(self.agents)
            joint_obs = np.concatenate((joint_obs, agents), axis=1)

        # Goal ids
        #
        # Sequence doesn't mean much
        # should be used to index a lookup table
        if self.obs_goals:
            goals = np.zeros((self.num_agents, 1))

            for i, agent in enumerate(self.agents):
                goals[i] = self.task_id[i]

            joint_obs = np.concatenate((joint_obs, goals), axis=1)

        return joint_obs

    def _get_img_obs(self, agent):
        obs = self.image[agent.row:agent.row + self.obs_height, agent.col:
                         agent.col + self.obs_width, :]

        if self.normalize:
            obs = obs / 255.0

        return obs

    def _on_goal(self, agent, success_map):
        cell_row = (agent.row + int(self.obs_height / 2)) // 10
        cell_col = (agent.col + int(self.obs_width / 2)) // 10

        assert cell_row < self.max_row_cells
        assert cell_col < self.max_col_cells

        return success_map[cell_row, cell_col]

    def is_success(self):
        return sum([
            self._on_goal(self.agents[i], self.success_maps[self.task_id[i]])
            for i in range(self.num_agents)
        ]) == self.num_agents

    # [TODO] this is potentially buggy
    def get_coverage(self):
        for agent in self.agents:
            r, c = agent.pos
            p = r * self.max_row + c
            if p not in self.visited_states:
                self.visited_states.append(p)
        return (1.0 * len(self.visited_states)) / self.max_states

    def _parse_task(self, task):
        self.task = task.split('.')[0]
        agent_goal_counts = [int(x) for x in task.split('.')[1].split(',')]
        assert sum(
            agent_goal_counts
        ) == self.num_agents, "No. of agents does not match goal definition"

        self.task_id = []
        for i in range(len(agent_goal_counts)):
            for j in range(agent_goal_counts[i]):
                self.task_id.append(i)

    def _move(self, agent, action):
        assert action in range(5)

        if action == 0:  # Move North
            agent.row = max(agent.row - self.step_size, 0)
        elif action == 1:  # Move East
            agent.col = min(agent.col + self.step_size, self.max_col)
        elif action == 2:  # Move South
            agent.row = min(agent.row + self.step_size, self.max_row)
        elif action == 3:  # Move West
            agent.col = max(agent.col - self.step_size, 0)
        else:
            pass

    def seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class Agent:
    def __init__(self, row, col):
        self.row = row
        self.col = col

    @property
    def pos(self):
        return self.row, self.col
