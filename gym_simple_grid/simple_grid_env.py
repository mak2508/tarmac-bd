import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
import random

class SimpleGridEnv(gym.Env):
    def __init__(self, n_agents=4, grid_size=10, max_steps=30, n_goals=5, 
                 obs_time_step=True, normalize=True, exploration_bonus=False):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.n_goals = n_goals
        self.max_steps = max_steps
        self.obs_time_step = obs_time_step
        self.normalize = normalize
        self.exploration_bonus = exploration_bonus
        
        if self.obs_time_step:
            self.observation_space = spaces.MultiDiscrete([grid_size, grid_size, 2, max_steps])
        else:
            self.observation_space = spaces.MultiDiscrete([grid_size, grid_size, 2])
            
        self.action_space = spaces.Discrete(5)
        
    def reset(self, agent_positions=None, goal_positions=None):
        self.n_steps = 0
        
        if goal_positions is not None:
            self.goals = goal_positions
        else:
            self.goals = [(random.randint(0, self.grid_size-1), 
                           random.randint(0, self.grid_size-1)) 
                          for _ in range(self.n_goals)]
            
        if agent_positions is not None:
            self.agents = [Agent(*pos) for pos in agent_positions]
        else:
            self.agents = [Agent(random.randint(0, self.grid_size-1), 
                                 random.randint(0, self.grid_size-1)) 
                           for _ in range(self.n_agents)] 
            
        if self.exploration_bonus:
            self.visited_positions = {agent.pos for agent in self.agents}
        
        joint_obs = self._get_joint_obs()
        
        return joint_obs

    def step(self, joint_action):
        self.n_steps += 1
        
        for agent, action in zip(self.agents, joint_action):
            self._move(agent, action)
        
        joint_obs = self._get_joint_obs()
        reward = sum([self._on_goal(agent) for agent in self.agents]) / self.n_agents 
        done = (self.n_steps == self.max_steps)
        
        if self.exploration_bonus:
            for agent in self.agents:
                reward += 0.1 * self._on_new_position(agent) / self.n_agents
                self.visited_positions.add(agent.pos)
        
        # Alternative reward formulation
        #reward = sum([self._on_goal(agent) for agent in self.agents]) / self.n_agents - 1.0
        #done = ((reward == 0.0) or (self.n_steps == self.max_steps))
            
        return joint_obs, reward, done, {}
   
    def render(self, save_dir=""):
        grid = np.zeros((self.grid_size, self.grid_size))
        
        for goal in self.goals:
            grid[goal] = 1
           
        if self.exploration_bonus:
            for pos in self.visited_positions:
                grid[pos] = 0.5
        
        for agent in self.agents:
            grid[agent.pos] = 2
           
        plt.imshow(grid)
        if save_dir is not "":
            plt.savefig(save_dir + str(self.n_steps))
        plt.show()
    
    def _get_joint_obs(self):
        joint_obs = np.zeros((self.n_agents, 3))
        
        for i, agent in enumerate(self.agents):
            joint_obs[i, :2] = agent.pos
            joint_obs[i, 2] = self._on_goal(agent)
            
        if self.normalize:
            joint_obs[:, :2] = joint_obs[:, :2] / self.grid_size
            
        if self.obs_time_step:
            time_step = np.full((self.n_agents, 1), self.n_steps)
            if self.normalize:
                time_step = time_step / self.max_steps
            joint_obs = np.hstack((joint_obs, time_step))
        
        return joint_obs
    
    def _on_goal(self, agent):
        return agent.pos in self.goals
    
    def _on_new_position(self, agent):
        return agent.pos not in self.visited_positions
    
    def _move(self, agent, action):
        assert action in range(5)
        
        if action == 0: # Move North
            agent.row = max(agent.row - 1, 0)  
        elif action == 1: # Move East
            agent.col = min(agent.col + 1, self.grid_size - 1) 
        elif action == 2: # Move South
            agent.row = min(agent.row + 1, self.grid_size - 1)
        elif action == 3: # Move West
            agent.col = max(agent.col - 1, 0) 
        else:
            pass
        
    def seed(self, seed):
        random.seed(seed)

class Agent:
    def __init__(self, row, col):
        self.row = row
        self.col = col
    
    @property
    def pos(self):
        return self.row, self.col