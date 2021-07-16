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

def evalPolicy(policy, env, N=100, Tmax=100, distance_threshold=0.05, logger=None):
    final_rewards = []
    successes = [] 
    puck, arm = [], []
    puck_dist, arm_dist = [], []
    for _ in range(N):
        obs = env.reset()
        done = False
        state = obs["image_observation"]
        goal = obs["image_desired_goal"]
        t = 0
        while not done:
            action = policy.select_action(state, goal)
            next_obs, reward, _, _ = env.step(action) 
            last_reward = reward
            
            next_state = next_obs["image_observation"]
            state = next_state
            reward = - np.sqrt(np.power(np.array(reward).reshape(2, 2), 2).sum(-1)).max(-1)    
            done = reward > -distance_threshold or t >= Tmax
            t += 1
        final_rewards.append(reward)
        successes.append( 1.0 * (reward > -distance_threshold ))

        # Additional info about arm and puck
        arm.append(1.0*(np.sqrt(np.power(np.array(last_reward[:2]), 2).sum()) < 0.06) )
        puck.append( 1.0*(np.sqrt(np.power(np.array(last_reward[2:]), 2).sum()) < 0.06) )
        arm_dist.append( np.sqrt(np.power(np.array(last_reward[:2]), 2).sum()) )
        puck_dist.append(np.sqrt(np.power(np.array(last_reward[2:]), 2).sum()) )

    eval_reward, success_rate = np.mean(final_rewards), np.mean(successes)

    if logger is not None:
        logger.store(eval_reward=eval_reward, success_rate=success_rate)
        logger.store(puck=np.mean(puck), arm=np.mean(arm), puck_dist=np.mean(puck_dist), arm_dist=np.mean(arm_dist))

    return eval_reward, success_rate



def sample_and_preprocess_batch(replay_buffer, batch_size=256, distance_threshold=0.05, device=torch.device("cuda")):
    # Extract 
    batch = replay_buffer.random_batch(batch_size)
    state_batch         = batch["observations"]
    action_batch        = batch["actions"]
    next_state_batch    = batch["next_observations"]
    goal_batch          = batch["resampled_goals"]
    reward_batch        = batch["rewards"]
    done_batch          = batch["terminals"] 

    # Compute sparse rewards: -1 for all actions until the goal is reached
    reward_batch = - np.sqrt(np.power(np.array(reward_batch).reshape(-1, 2, 2), 2).sum(-1)).max(-1).reshape(-1, 1) 
    done_batch = 1.0* (reward_batch > -distance_threshold)
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
    parser.add_argument("--env",                default="Image84SawyerPushAndReachArenaTrainEnvBig-v0")
    parser.add_argument("--test_env",           default="Image84SawyerPushAndReachArenaTestEnvBig-v1")
    parser.add_argument("--epsilon",            default=1e-4, type=float)
    parser.add_argument("--replay_buffer_goals",default=0.5, type=float)
    parser.add_argument("--distance_threshold", default=0.05, type=float)
    parser.add_argument("--start_timesteps",    default=1e4, type=int) 
    parser.add_argument("--eval_freq",          default=1e3, type=int)
    parser.add_argument("--max_timesteps",      default=5e6, type=int)
    parser.add_argument("--max_episode_length", default=100, type=int)
    parser.add_argument("--batch_size",         default=256, type=int)
    parser.add_argument("--replay_buffer_size", default=1e5, type=int)
    parser.add_argument("--n_eval",             default=20, type=int)
    parser.add_argument("--device",             default="cuda")
    parser.add_argument("--seed",               default=42, type=int)
    parser.add_argument("--exp_name",           default="RIS_sawyer")
    parser.add_argument("--alpha",              default=0.1, type=float)
    parser.add_argument("--Lambda",             default=0.1, type=float)
    parser.add_argument("--h_lr",               default=1e-4, type=float)
    parser.add_argument("--q_lr",               default=1e-3, type=float)
    parser.add_argument("--pi_lr",              default=1e-4, type=float)
    parser.add_argument("--enc_lr",             default=1e-4, type=float)
    parser.add_argument("--state_dim",          default=16, type=int)

    parser.add_argument('--log_loss', dest='log_loss', action='store_true')
    parser.add_argument('--no-log_loss', dest='log_loss', action='store_false')
    parser.set_defaults(log_loss=True)
    args = parser.parse_args()
    print(args)

    register_mujoco_envs()

    # Set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env         = gym.make(args.env)
    test_env    = gym.make(args.test_env)

    action_dim = env.action_space.shape[0]
    state_dim = args.state_dim 

    # Create logger
    logger = Logger(vars(args))
    
    # Initialize policy
    policy = RIS(state_dim=state_dim, action_dim=action_dim, image_env=True, alpha=args.alpha, Lambda=args.Lambda, epsilon=args.epsilon, h_lr=args.h_lr, q_lr=args.q_lr, pi_lr=args.pi_lr, enc_lr=args.enc_lr, device=args.device, logger=logger if args.log_loss else None)


   
    # Initialize replay buffer and path_builder
    replay_buffer = HERReplayBuffer(
        max_size=args.replay_buffer_size,
        env=env,
        fraction_resampled_goals_are_replay_buffer_goals = args.replay_buffer_goals, 
        ob_keys_to_save     =["state_achieved_goal", "state_desired_goal"],
        desired_goal_keys   =["image_desired_goal", "state_desired_goal"],
        observation_key     = 'image_observation',
        desired_goal_key    = 'image_desired_goal',
        achieved_goal_key   = 'image_achieved_goal',
        vectorized          = True
    )
    path_builder = PathBuilder()


    # Initialize environment
    obs = env.reset()
    done = False
    state = obs["image_observation"]
    goal = obs["image_desired_goal"]
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
        next_obs, reward, _, _ = env.step(action) 
        
        next_state = next_obs["image_observation"]
        done = - np.sqrt(np.power(np.array(reward).reshape(2, 2), 2).sum(-1)).max(-1) > -args.distance_threshold 

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
            logger.store(t=t, reward=reward)		

            # Reset environment
            obs = env.reset()
            done = False
            state = obs["image_observation"]
            goal = obs["image_desired_goal"]
            episode_timesteps = 0
            episode_num += 1 

        if (t + 1) % args.eval_freq == 0 and t >= args.start_timesteps:
            # Eval policy
            eval_reward, success_rate = evalPolicy(
                policy, test_env, 
                N=args.n_eval,
                Tmax=args.max_episode_length, 
                distance_threshold=args.distance_threshold,
                logger = logger
            )
            print("RIS t={} | {}".format(t+1, logger))

            # Save results
            folder = "results/{}/RIS/{}/".format(args.env, args.exp_name)
            if not os.path.isdir(folder):
                os.mkdir(folder)
            logger.save(folder + "log.pkl")
            policy.save(folder)

