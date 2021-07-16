import os

import torch
import numpy as np
import random
import argparse

import gym
from multiworld.envs.mujoco import register_custom_envs as register_mujoco_envs

import time

from utils.logger import Logger
from RIS import RIS
from HER import HERReplayBuffer, PathBuilder


def evalPolicy(policy, env, N=100, Tmax=100, distance_threshold=0.5, logger=None):
    final_distance = []
    successes = [] 

    for _ in range(N):
        obs = env.reset()
        done = False
        state = obs["observation"]
        goal = obs["desired_goal"]
        t = 0
        while not done:
            action = policy.select_action(state, goal)
            next_obs, _, _, status = env.step(action) 
            next_state = next_obs["observation"]
            state = next_state
            done = status["xy-distance"] < distance_threshold or t >= Tmax
            t += 1

        final_distance.append(status["xy-distance"])
        successes.append( 1.0 * (status["xy-distance"] < distance_threshold ))

    eval_distance, success_rate =np.mean(final_distance), np.mean(successes)
    if logger is not None:
        logger.store(eval_distance=eval_distance, success_rate=success_rate)
    return eval_distance, success_rate

def sample_and_preprocess_batch(replay_buffer, batch_size=1024, distance_threshold=0.5, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"] 
    
    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = - np.sqrt(np.power(np.array(next_state_batch - goal_batch)[:, :2], 2).sum(-1, keepdims=True))
    done_batch   = 1.0 * (reward_batch > -distance_threshold) 
    reward_batch = - np.ones_like(done_batch)

    # Convert to Pytorch
    state_batch         = torch.FloatTensor(state_batch).to(device)
    action_batch        = torch.FloatTensor(action_batch).to(device)
    reward_batch        = torch.FloatTensor(reward_batch).to(device)
    next_state_batch    = torch.FloatTensor(next_state_batch).to(device)
    done_batch          = torch.FloatTensor(done_batch).to(device)
    goal_batch          = torch.FloatTensor(goal_batch).to(device)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch

if __name__ == "__main__":	
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name",                default="AntU")
    parser.add_argument("--distance_threshold", default=0.5, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=1e3, type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--max_episode_length", default=600, type=int)
    parser.add_argument("--batch_size",         default=2048, type=int)
    parser.add_argument("--replay_buffer_size", default=1e6, type=int)
    parser.add_argument("--n_eval",             default=5, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="RIS_ant")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-3, type=float)
    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    # select environment
    if args.env_name == "AntU":
        train_env_name = "AntULongTrainEnv-v0"
        test_env_name = "AntULongTestEnv-v0"
    elif args.env_name == "AntFb":
        train_env_name = "AntFbMedTrainEnv-v1"
        test_env_name = "AntFbMedTestEnv-v1"
    elif args.env_name == "AntMaze":
        train_env_name = "AntMazeMedTrainEnv-v1"
        test_env_name = "AntMazeMedTestEnv-v1"
    elif args.env_name == "AntFg":
        train_env_name = "AntFgMedTrainEnv-v1"
        test_env_name = "AntFgMedTestEnv-v1"
    print("Environments: ", train_env_name, test_env_name)

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment
    register_mujoco_envs()
    vectorized = True
    goal_dim = 17 
    env         = gym.make(train_env_name)
    test_env    = gym.make(test_env_name)
    action_dim = env.action_space.shape[0]
    state_dim = 31 

    # Create logger
    logger = Logger(vars(args))
    policy = RIS(state_dim=state_dim, action_dim=action_dim, alpha=args.alpha, Lambda=args.Lambda, h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, device=args.device, logger=logger if args.log_loss else None)
   
    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_goals_are_rollout_goals = 0.2,
        fraction_resampled_goals_are_env_goals = 0.0,
        fraction_resampled_goals_are_replay_buffer_goals = 0.5,
        ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys   =["desired_goal", "state_desired_goal"],
        observation_key     = 'observation',
        desired_goal_key    = 'desired_goal',
        achieved_goal_key   = 'achieved_goal',
        vectorized          = vectorized 
    )
    path_builder = PathBuilder()

    # Initialize environment
    obs = env.reset()
    done = False
    state = obs["observation"]
    goal = obs["desired_goal"]
    episode_timesteps = 0
    episode_num = 0 

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        # Select action
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.select_action(state, goal)

        # Perform action
        next_obs, reward, _, status = env.step(action) 
        
        next_state = next_obs["observation"]
        done = status["xy-distance"][0] < args.distance_threshold

        path_builder.add_all(
            observations=obs,
            actions=action,
            rewards=reward,
            next_observations=next_obs,
            terminals=[1.0*done]
        )

        state = next_state
        obs = next_obs


        # Train agent after collecting enough data
        if t >= args.batch_size and t >= args.start_timesteps:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch = sample_and_preprocess_batch(
                replay_buffer, 
                batch_size=args.batch_size, 
                distance_threshold=args.distance_threshold,
                device=args.device
            )
            # Sample subgoal candidates uniformly in the replay buffer
            subgoal_batch = torch.FloatTensor(replay_buffer.random_state_batch(args.batch_size)).to(args.device)
            policy.train(state_batch, action_batch, reward_batch, next_state_batch, done_batch, goal_batch, subgoal_batch)


        if done or episode_timesteps >= args.max_episode_length: 
            # Add path to replay buffer and reset path builder
            replay_buffer.add_path(path_builder.get_all_stacked())
            path_builder = PathBuilder()
            logger.store(t=t, distance=status["xy-distance"][0]) 

            obs = env.reset()
            done = False
            state = obs["observation"]
            goal = obs["desired_goal"]
            episode_timesteps = 0
            episode_num += 1 

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
            evalPolicy(
                policy, test_env, 
                N=args.n_eval,
                Tmax=args.max_episode_length, 
                distance_threshold=args.distance_threshold,
                logger = logger
            )
            print("RIS | {}".format(logger))

            # Save results
            folder = "results/{}/RIS/{}/".format(train_env_name, args.exp_name)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            logger.save(folder + "log.pkl")
            policy.save(folder)
