import os
import copy
import glob
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env_multi_agent
from model import MultiAgentPolicy
from storage import MultiAgentRolloutStorage
from arguments import traffic_env_config

from gym.envs.registration import register

from tensorboardX import SummaryWriter
import shutil

import algo

import pdb


def main(args):

    for arg in vars(args):
        print('{:<30}: {}'.format(arg, getattr(args, arg)))

    if args.log == True:
        # Tensorboard summary writer
        writer = SummaryWriter(args.tensorboard_dir)

    # Re-register env to add custom args
    if args.env_name == 'ShapesEnv-v0':

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

    elif args.env_name == 'TrafficEnv-v0':

        args.num_agents = traffic_env_config[args.difficulty]['num_agents']
        args.num_steps = args.traffic_num_steps

        register(
            id='CustomTrafficEnv-v0',
            entry_point='gym_traffic.traffic_env:TrafficEnv',
            kwargs={
                'num_steps': args.traffic_num_steps,
                'difficulty': traffic_env_config[args.difficulty]['difficulty'],
                'dim': traffic_env_config[args.difficulty]['dim'],
                'vision': traffic_env_config[args.difficulty]['vision'],
                'num_agents': traffic_env_config[args.difficulty]['num_agents'],
                'add_rate_min': traffic_env_config[args.difficulty]['add_rate'],
                'add_rate_max': traffic_env_config[args.difficulty]['add_rate']
            })

        # Parallel environments
        print('Setting up {} parallel envs'.format(args.num_processes))
        envs = [
            make_env_multi_agent('CustomTrafficEnv-v0', args.seed, i)
            for i in range(args.num_processes)
        ]

        envs = SubprocVecEnv(envs) if (args.num_processes >
                                   1) else DummyVecEnv(envs)
        obs_shape = envs.observation_space.shape

    elif args.env_name == 'House3DEnv-v0':

        register(
            id='CustomHouse3DEnv-v0',
            entry_point='gym_house3d.house3d_env:House3DEnv',
            kwargs={
                'num_agents': args.num_agents,
                'num_steps': args.num_steps,
                'fix_env': args.fix_env,
                'fix_spawn': args.fix_spawn
            })

        # Parallel environments
        print('Setting up {} parallel envs'.format(args.num_processes))
        envs = [
            make_env_multi_agent('CustomHouse3DEnv-v0', args.seed, i)
            for i in range(args.num_processes)
        ]

        envs = SubprocVecEnv(envs) if (args.num_processes >
                                   1) else DummyVecEnv(envs)
        obs_shape = envs.observation_space.shape

    # Model holding weights
    print('Setting up model')
    actor_critic = MultiAgentPolicy(
        args.num_agents, obs_shape[0], envs.action_space.n,
        args.recurrent_policy, args.state_size, args.comm_size,
        args.comm_mode, args.comm_num_hops,
        use_cnn=args.use_cnn, env=args.env_name)

    if args.cuda:
        actor_critic.cuda()

    # Algorithm holding optimizer
    agent = algo.A2C_ACKTR(
        actor_critic,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        alpha=args.alpha,
        distributed=args.distributed,
        max_grad_norm=args.max_grad_norm)

    # Storage
    rollouts = MultiAgentRolloutStorage(
        args.num_agents, obs_shape, args.num_steps, args.num_processes,
        actor_critic.state_size, actor_critic.comm_size)

    obs = envs.reset()

    rollouts.observations[0].copy_(torch.from_numpy(obs))

    if args.cuda:
        rollouts.cuda()

    # Train
    print('Training')

    start = time.time()
    for j in range(args.num_updates):

        # Variables used to compute average rewards for all processes
        episode_rewards = torch.zeros([args.num_processes, args.num_agents, 1])
        final_rewards = torch.zeros([args.num_processes, args.num_agents, 1])

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, actions, actions_log_probs, states, communications, aux = actor_critic.act(
                    rollouts.observations[step], rollouts.states[step],
                    rollouts.communications[step], rollouts.masks[step])
            cpu_actions = actions.squeeze(2).cpu().numpy()

            # Observe reward and next obs
            obs, reward, done, info = envs.step(cpu_actions)

            obs = torch.from_numpy(obs)
            # reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            reward = torch.from_numpy(np.expand_dims(reward, 2)).float()
            episode_rewards += reward

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            masks = masks.unsqueeze(1).expand(args.num_processes, args.num_agents, 1)

            # When an env is done, mask is set to 0
            # final_rewards is just for logging,
            # and stores reward sum at final time-step per env
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            obs = torch.tensor(obs)
            if args.cuda:
                masks = masks.cuda()
                obs = obs.cuda()

            # Store experience
            rollouts.insert(obs, states, actions, actions_log_probs, value,
                            reward, masks, communications)

        if 'success' in info[0]:
            success = [x['success'] for x in info]
        if 'coverage' in info[0]:
            coverage = [x['coverage'] for x in info]

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.observations[-1], rollouts.states[-1],
                rollouts.communications[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.tau)
        value_loss, action_loss, dist_entropy = agent.update_multi_agent(
            rollouts)
        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "" and args.log == True:
            # Save model
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()
            torch.save(save_model,
                       os.path.join(args.save_dir, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            # Print training statistics
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            if args.env_name == 'ShapesEnv-v0':
                print(
                    "[t:%010d][fps:%05d][success:%.02f][coverage:%.02f][rwd:%.02f/%.02f][entropy:%.05f][value_loss:%.05f][policy_loss:%.05f]"
                    % (total_num_steps, int(total_num_steps / (end - start)),
                       np.mean(success) * 100, np.mean(coverage) * 100,
                       final_rewards.mean(), final_rewards.median(), dist_entropy,
                       value_loss, action_loss))
            elif args.env_name == 'TrafficEnv-v0':
                print(
                "[t:%010d][fps:%05d][success:%.02f][rwd:%.02f/%.02f][entropy:%.05f][value_loss:%.05f][policy_loss:%.05f]"
                % (total_num_steps, int(total_num_steps / (end - start)),
                    np.mean(success) * 100,
                   final_rewards.mean(), final_rewards.median(), dist_entropy,
                   value_loss, action_loss))
            elif args.env_name == 'House3DEnv-v0':
                print(
                "[t:%010d][fps:%05d][rwd:%.02f/%.02f][entropy:%.05f][value_loss:%.05f][policy_loss:%.05f]"
                % (total_num_steps, int(total_num_steps / (end - start)),
                   final_rewards.mean(), final_rewards.median(), dist_entropy,
                   value_loss, action_loss))
            else:
                raise NotImplementedError

            if args.log == True:
                # Save training statistics to tensorboard
                if args.env_name == 'ShapesEnv-v0':
                    writer.add_scalar("Success",
                                      np.mean(success) * 100, total_num_steps)
                    writer.add_scalar("Coverage",
                                      np.mean(coverage) * 100, total_num_steps)
                    writer.add_scalar("RewardMean", final_rewards.mean(),
                                      total_num_steps)
                    writer.add_scalar("Entropy", dist_entropy, total_num_steps)
                    writer.add_scalar("ValueLoss", value_loss, total_num_steps)
                    writer.add_scalar("ActorLoss", action_loss, total_num_steps)
                elif args.env_name == 'TrafficEnv-v0':
                    writer.add_scalar("Success",
                                      np.mean(success) * 100, total_num_steps)
                    writer.add_scalar("RewardMean", final_rewards.mean(),
                                      total_num_steps)
                    writer.add_scalar("Entropy", dist_entropy, total_num_steps)
                    writer.add_scalar("ValueLoss", value_loss, total_num_steps)
                    writer.add_scalar("ActorLoss", action_loss, total_num_steps)
                elif args.env_name == 'House3DEnv-v0':
                    writer.add_scalar("RewardMean", final_rewards.mean(),
                                      total_num_steps)
                    writer.add_scalar("Entropy", dist_entropy, total_num_steps)
                    writer.add_scalar("ValueLoss", value_loss, total_num_steps)
                    writer.add_scalar("ActorLoss", action_loss, total_num_steps)
                else:
                    raise NotImplementedError

    if args.log == True:
        writer.close()