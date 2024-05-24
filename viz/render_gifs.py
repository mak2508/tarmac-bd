import matplotlib
matplotlib.use('Agg')

import glob
import os, sys
import shutil
from tqdm import tqdm

sys.path.insert(0, './')

import json
import numpy as np

import torch

import gym
import gym_simple_grid
import gym_shapes
import gym_traffic

from gym.envs.registration import register

import moviepy.editor as mpy

import multiprocessing
from multiprocessing import Process, Pool

from arguments import traffic_env_config

import yaml
import argparse

import pdb


def render_gif_for_trajectory_traffic(model, img_id, config):

    hparam_str = model

    # Retrieve trained model
    path = os.path.join(config['dir']['base'], hparam_str, "TrafficEnv-v0.pt")
    actor_critic = torch.load(path, map_location={'cuda:0': 'cpu'})

    env = gym.make('CustomTrafficEnv-v0')
    env.seed(123+img_id)

    gif_dir = os.path.join(config['dir']['gif'], hparam_str)
    try:
        os.makedirs(gif_dir)
    except:
        pass

    temp_dir = os.path.join(config['dir']['temp'], hparam_str, str(img_id))
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    try:
        os.makedirs(temp_dir)
    except:
        pass

    # Run episode and save png images with .render()
    obs = env.reset()
    states = torch.zeros(1, actor_critic.n_agents, actor_critic.state_size)
    communications = torch.zeros(1, actor_critic.n_agents,
                                 actor_critic.comm_size)
    masks = torch.ones(1, 1)

    aux = {}
    obs = torch.FloatTensor(obs).unsqueeze(0)
    value, actions, actions_log_probs, states, communications, aux = actor_critic.act(
        obs, states, communications, masks)
    actions = actions[0, :, 0].cpu().numpy()
    obs, reward, done, _ = env.step(actions)
    env.render(save_dir=temp_dir, aux=aux)

    while True:
        obs = torch.FloatTensor(obs).unsqueeze(0)

        value, actions, actions_log_probs, states, communications, aux = actor_critic.act(
            obs, states, communications, masks)
        actions = actions[0, :, 0].cpu().numpy()

        obs, reward, done, _ = env.step(actions)

        env.render(save_dir=temp_dir, aux=aux)

        if done:
            break

    # Create gif
    gif_name = os.path.join(gif_dir, str(img_id) + '.mp4')
    file_list = glob.glob(temp_dir + '/*.png')
    file_list.sort()
    clip = mpy.ImageSequenceClip(file_list, fps=config['render']['fps'])
    clip.write_videofile(gif_name, fps=config['render']['fps'])

    print('Saved %s' % gif_name)


def render_gif_for_trajectory_shapes(model, img_id, config):

    hparam_str = model

    # Retrieve trained model
    path = os.path.join(config['dir']['base'], hparam_str, "ShapesEnv-v0.pt")
    actor_critic = torch.load(path, map_location={'cuda:0': 'cpu'})

    env = gym.make('CustomShapesEnv-v0')

    gif_dir = os.path.join(config['dir']['gif'], hparam_str)
    try:
        os.makedirs(gif_dir)
    except:
        pass

    temp_dir = os.path.join(config['dir']['temp'], hparam_str, str(img_id))
    try:
        shutil.rmtree(temp_dir)
    except:
        pass

    try:
        os.makedirs(temp_dir)
    except:
        pass

    # Run episode and save png images with .render()
    obs = env.reset(img_idx=img_id)
    states = torch.zeros(1, actor_critic.n_agents, actor_critic.state_size)
    communications = torch.zeros(1, actor_critic.n_agents,
                                 actor_critic.comm_size)
    masks = torch.ones(1, 1)

    aux = {}
    if 'att' in model:
        aux['p_attn'] = torch.zeros(actor_critic.n_agents,
                                    actor_critic.n_agents)

    env.render(save_dir=temp_dir, aux=aux)

    while True:
        obs = torch.FloatTensor(obs).unsqueeze(0)

        value, actions, actions_log_probs, states, communications, aux = actor_critic.act(
            obs, states, communications, masks)
        actions = actions[0, :, 0].cpu().numpy()

        obs, reward, done, _ = env.step(actions)

        env.render(save_dir=temp_dir, aux=aux)

        if done:
            break

    # Create gif
    gif_name = os.path.join(gif_dir, str(img_id) + '.mp4')
    file_list = glob.glob(temp_dir + '/*.png')
    file_list.sort()
    clip = mpy.ImageSequenceClip(file_list, fps=config['render']['fps'])
    clip.write_videofile(gif_name, fps=config['render']['fps'])

    print('Saved %s' % gif_name)


if __name__ == '__main__':

    # Load config
    parser = argparse.ArgumentParser(description='rendering gifs')
    parser.add_argument('--cfg', default=None, help='path to yaml config')
    args = parser.parse_args()

    config = yaml.load(open(args.cfg, 'r'))

    if 'env' not in config or config['env'] == 'ShapesEnv':

        # Re-register env to add custom args
        register(
            id='CustomShapesEnv-v0',
            entry_point='gym_shapes.shapes_env:ShapesEnv',
            kwargs={
                'data_dir': config['dir']['data'],
                'task': config['task']['goal'],
                'num_agents': config['task']['num_agents'],
                'step_size': config['task']['step_size'],
                'num_steps': config['task']['num_steps'],
                'obs_height': config['task']['obs_height'],
                'obs_width': config['task']['obs_width'],
                'fix_spawn': config['task']['fix_spawn'],
                'fix_image': config['task']['fix_image']
            })

        img_ids = [x for x in range(20)]

        pool = Pool(processes=multiprocessing.cpu_count() - 1 or 1)

        for i in tqdm(range(len(img_ids))):
            # render_gif_for_trajectory(config['dir']['model'], img_ids[i], config, )
            pool.apply_async(
                render_gif_for_trajectory_shapes,
                args=(
                    config['dir']['model'],
                    img_ids[i],
                    config,
                ))

        pool.close()
        pool.join()

    elif config['env'] == 'TrafficEnv':

        # Re-register env to add custom args
        register(
            id='CustomTrafficEnv-v0',
            entry_point='gym_traffic.traffic_env:TrafficEnv',
            kwargs={
                'num_steps': config['task']['num_steps'],
                'difficulty': config['task']['difficulty'],
                'dim': traffic_env_config[config['task']['difficulty']]['dim'],
                'vision': traffic_env_config[config['task']['difficulty']]['vision'],
                'add_rate_min': 0.50,
                'add_rate_max': 0.50,
                'num_agents': traffic_env_config[config['task']['difficulty']]['num_agents']
            })

        pool = Pool(processes=multiprocessing.cpu_count() - 1 or 1)

        for i in range(10):
            # render_gif_for_trajectory_traffic(config['dir']['model'], i, config, )
            pool.apply_async(
                render_gif_for_trajectory_traffic,
                args=(
                    config['dir']['model'],
                    i,
                    config,
                ))

        pool.close()
        pool.join()
