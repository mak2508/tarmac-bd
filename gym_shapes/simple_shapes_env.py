import os
import json
import copy
import random

import matplotlib.pyplot as plt
import numpy as np

import gym
from gym import spaces

class SimpleShapesEnv(gym.Env):
    def __init__(
            self,
            data_dir='gym_shapes/data/shapes/shapes_06_05_08:34',
            split='train.large', # train.med, train.small, train.tiny, val, test
            n_agents=4,
            obs_height=5,
            obs_width=5,
            step_size=2,
            max_steps=30,
            normalize=True,
            obs_coordinates=True,
            obs_time_step=True):
        
        self.n_agents = n_agents
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.step_size = step_size
        self.max_steps = max_steps
        self.normalize = normalize
        self.obs_coordinates = obs_coordinates
        self.obs_time_step = obs_time_step
        
        # Load data
        self.data_root = os.path.join(data_dir, split)
        self.data = np.load(self.data_root + '.input.npy')[:, :, :, ::-1]
        
        self.n_imgs = self.data.shape[0]
        self.img_shape = self.data[0].shape
        self.img_obs_shape = (self.obs_height, self.obs_width, self.img_shape[2])
        
        self.max_row = self.img_shape[0] - self.obs_height
        self.max_col = self.img_shape[1] - self.obs_width   
        
        # Gym parameters
        self._action_space = spaces.Discrete(5) 
        
        len_obs = self.img_obs_shape[0] * self.img_obs_shape[1] * self.img_obs_shape[2]
        if self.obs_coordinates:
            len_obs += 2
        if self.obs_time_step:
            len_obs += 1
        self._observation_space = spaces.Box(low=0.0, high=1.0, shape=(len_obs,), dtype=np.float32)
    
    def reset(self, agent_positions=None, goal_position=None, img_idx=9):
        self.n_steps = 0
        
        self.image = copy.deepcopy(self.data[img_idx]) 
        
        if goal_position is not None:
            self.goal = Agent(*goal_position)
        else:
            self.goal = Agent(random.randint(0, self.max_row), 
                              random.randint(0, self.max_col))
        
        self.image[self.goal.row:self.goal.row + self.obs_height,
                   self.goal.col:self.goal.col + self.obs_width, :] = 255
        
        if agent_positions is not None:
            self.agents = [Agent(*pos) for pos in agent_positions]
        else:
            self.agents = [Agent(random.randint(0, self.max_row), 
                                 random.randint(0, self.max_col)) 
                           for _ in range(self.n_agents)] 
        
        joint_obs = self._get_joint_obs()
        
        return joint_obs

    def step(self, joint_action):
        self.n_steps += 1
        
        for agent, action in zip(self.agents, joint_action):
            self._move(agent, action)
        
        joint_obs = self._get_joint_obs()
        reward = sum([self._on_goal(agent) for agent in self.agents]) / self.n_agents
        done = (self.n_steps == self.max_steps)
            
        return joint_obs, reward, done, {}
    
    def render(self, save_dir=""):
        visions = np.zeros(self.img_shape, dtype=int)
        positions = np.zeros(self.img_shape[:2], dtype=int)
        
        for agent in self.agents:
            r, c = agent.pos
            h, w = self.obs_height, self.obs_width
            visions[r:r + h, c:c + w, :] = self.image[r:r + h, c:c + w, :]
            positions[r:r + h, c:c + w] = 1
        
        fig = plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(self.image)
        plt.subplot(1, 3, 2)
        plt.imshow(visions)
        plt.subplot(1, 3, 3)
        plt.imshow(positions)
        if save_dir is not "":
            plt.savefig(save_dir + str(self.n_steps))
        plt.show()
        
    def _get_joint_obs(self):
        # Image observations
        img_obs = np.zeros((self.n_agents, *self.img_obs_shape))
        
        for i, agent in enumerate(self.agents):
            img_obs[i] = self._get_img_obs(agent)
            
        joint_obs = img_obs.reshape(self.n_agents, -1)

        # Coordinate observations
        if self.obs_coordinates:
            coordinates = np.zeros((self.n_agents, 2))
            
            for i, agent in enumerate(self.agents):
                coordinates[i] = np.array(agent.pos)
                
            if self.normalize:
                coordinates[:, 0] = coordinates[:, 0] / self.max_row
                coordinates[:, 1] = coordinates[:, 1] / self.max_col
            
            joint_obs = np.concatenate((joint_obs, coordinates), axis=1)
            
        # Time step observation
        if self.obs_time_step:
            time_step = np.full((self.n_agents, 1), self.n_steps)
            
            if self.normalize:
                time_step = time_step / self.max_steps
                
            joint_obs = np.concatenate((joint_obs, time_step), axis=1)
        
        return joint_obs
    
    def _get_img_obs(self, agent):
        obs = self.image[agent.row:agent.row + self.obs_height, 
                          agent.col:agent.col + self.obs_width, :]
        
        if self.normalize:
            obs = obs / 255.0
            
        return obs
    
    def _on_goal(self, agent):
        row_on_goal = agent.row - int(self.obs_height / 2) <= self.goal.row <= agent.row + int(self.obs_height / 2)
        col_on_goal = agent.col - int(self.obs_width / 2) <= self.goal.col <= agent.col + int(self.obs_width / 2)
        return row_on_goal and col_on_goal
    
    def _move(self, agent, action):
        assert action in range(5)
        
        if action == 0: # Move North
            agent.row = max(agent.row - self.step_size, 0)  
        elif action == 1: # Move East
            agent.col = min(agent.col + self.step_size, self.max_col) 
        elif action == 2: # Move South
            agent.row = min(agent.row + self.step_size, self.max_row)
        elif action == 3: # Move West
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