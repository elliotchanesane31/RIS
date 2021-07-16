"""
Credits: https://github.com/snasiriany/leap
"""

import numpy as np
from gym.spaces import Dict

from multiworld.core.image_env import unormalize_image, normalize_image

class PathBuilder(dict):
    """
    Usage:
    ```
    path_builder = PathBuilder()
    path.add_sample(
        observations=1,
        actions=2,
        next_observations=3,
        ...
    )
    path.add_sample(
        observations=4,
        actions=5,
        next_observations=6,
        ...
    )
    path = path_builder.get_all_stacked()
    path['observations']
    # output: [1, 4]
    path['actions']
    # output: [2, 5]
    ```
    Note that the key should be "actions" and not "action" since the
    resulting dictionary will have those keys.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        output_dict = dict()
        for k, v in self.items():
            output_dict[k] = stack_list(v)
        return output_dict

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        return lst
    else:
        return np.array(lst)


class HERReplayBuffer():
    """
    Save goals from the same trajectory into the replay buffer.
    Only add_path is implemented.
    Implementation details:
     - Every sample from [0, self._size] will be valid.
     - Observation and next observation are saved separately. It's a memory
       inefficient to save the observations twice, but it makes the code
       *much* easier since you no longer have to worry about termination
       conditions.
    """

    def __init__(
            self,
            max_size,
            env,
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.0,
            fraction_resampled_goals_are_replay_buffer_goals=0.5,
            ob_keys_to_save=None,
            internal_keys=None,
            desired_goal_keys=None,
            goal_keys=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
            achieved_goal_key='achieved_goal',
            vectorized=False,
    ):
        """
        :param max_size:
        :param env:
        :param fraction_goals_rollout_goals: Default, no her
        :param fraction_resampled_goals_env_goals:  of the resampled
        goals, what fraction are sampled from the env. Reset are sampled from future.
        :param ob_keys_to_save: List of keys to save
        """
        self.env = env
        self.ob_spaces = self.env.observation_space.spaces

        if ob_keys_to_save is not None:
            ob_keys_to_save = list(ob_keys_to_save)
        else:
            ob_keys_to_save = []
        if internal_keys is None:
            internal_keys = []
        self.internal_keys = internal_keys
        if goal_keys is None:
            goal_keys = [desired_goal_key]
        self.goal_keys = goal_keys
        if desired_goal_keys is None:
            desired_goal_keys = [desired_goal_key]
        self.desired_goal_keys = desired_goal_keys
        if desired_goal_key not in self.goal_keys:
            self.goal_keys.append(desired_goal_key)

        assert isinstance(env.observation_space, Dict)
        max_size = int(max_size)
        self.max_size = max_size
        self.fraction_goals_rollout_goals = fraction_goals_are_rollout_goals
        self.fraction_resampled_goals_env_goals = fraction_resampled_goals_are_env_goals
        self.fraction_resampled_goals_replay_buffer_goals = fraction_resampled_goals_are_replay_buffer_goals
        self.ob_keys_to_save = ob_keys_to_save
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key
        self.achieved_goal_key = achieved_goal_key

        self._action_dim = env.action_space.low.size
        self._actions = np.ones((max_size, self._action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self.vectorized = vectorized
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.ones((max_size, 1), dtype='uint8')
        # self._obs[key][i] is the value of observation[key] at time i
        self._obs = {}
        self._next_obs = {}
        for key in [observation_key, desired_goal_key, achieved_goal_key]:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)

        for key in ob_keys_to_save + internal_keys:
            assert key in self.ob_spaces, \
                "Key not found in the observation space: %s" % key
            type = np.float64
            if key.startswith('image'):
                type = np.uint8
            self._obs[key] = np.ones(
                (max_size, self.ob_spaces[key].low.size), dtype=type)
            self._next_obs[key] = np.ones(
                (max_size, self.ob_spaces[key].low.size), dtype=type)

        self._top = 0
        self._size = 0

        # Let j be any index in self._idx_to_future_obs_idx[i]
        # Then self._next_obs[j] is a valid next observation for observation i
        self._idx_to_future_obs_idx = [None] * max_size

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        raise NotImplementedError("Only use add_path")

    def terminate_episode(self):
        pass

    def num_steps_can_sample(self):
        return self._size

    def add_path(self, path):
        obs = path["observations"]
        actions = path["actions"]
        rewards = path["rewards"]
        next_obs = path["next_observations"]
        terminals = path["terminals"]
        path_len = len(rewards)

        actions = flatten_n(actions)
        obs = flatten_dict(obs, self.ob_keys_to_save + self.internal_keys)
        next_obs = flatten_dict(next_obs, self.ob_keys_to_save + self.internal_keys)
        obs = preprocess_obs_dict(obs)
        next_obs = preprocess_obs_dict(next_obs)

        if self._top + path_len >= self.max_size:
            num_pre_wrap_steps = self.max_size - self._top
            # numpy slice
            pre_wrap_buffer_slice = np.s_[
                                    self._top:self._top + num_pre_wrap_steps, :
                                    ]
            pre_wrap_path_slice = np.s_[0:num_pre_wrap_steps, :]

            num_post_wrap_steps = path_len - num_pre_wrap_steps
            post_wrap_buffer_slice = slice(0, num_post_wrap_steps)
            post_wrap_path_slice = slice(num_pre_wrap_steps, path_len)
            for buffer_slice, path_slice in [
                (pre_wrap_buffer_slice, pre_wrap_path_slice),
                (post_wrap_buffer_slice, post_wrap_path_slice),
            ]:
                self._actions[buffer_slice] = actions[path_slice]
                self._terminals[buffer_slice] = terminals[path_slice]
                for key in self.ob_keys_to_save + self.internal_keys:
                    self._obs[key][buffer_slice] = obs[key][path_slice]
                    self._next_obs[key][buffer_slice] = next_obs[key][path_slice]
            # Pointers from before the wrap
            for i in range(self._top, self.max_size):
                self._idx_to_future_obs_idx[i] = np.hstack((
                    # Pre-wrap indices
                    np.arange(i, self.max_size),
                    # Post-wrap indices
                    np.arange(0, num_post_wrap_steps)
                ))
            # Pointers after the wrap
            for i in range(0, num_post_wrap_steps):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i,
                    num_post_wrap_steps,
                )
        else:
            slc = np.s_[self._top:self._top + path_len, :]
            self._actions[slc] = actions
            self._terminals[slc] = terminals
            for key in self.ob_keys_to_save + self.internal_keys:
                self._obs[key][slc] = obs[key]
                self._next_obs[key][slc] = next_obs[key]
            for i in range(self._top, self._top + path_len):
                self._idx_to_future_obs_idx[i] = np.arange(
                    i, self._top + path_len
                )
        self._top = (self._top + path_len) % self.max_size
        self._size = min(self._size + path_len, self.max_size)

    def _sample_indices(self, batch_size):
        return np.random.randint(0, self._size, batch_size)

    def random_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        resampled_goals = {
            desired_goal_key: self._next_obs[desired_goal_key][indices] for desired_goal_key in self.desired_goal_keys
        }
        num_rollout_goals = int(
            batch_size * self.fraction_goals_rollout_goals
        )
        num_env_goals = int(
            batch_size * (1 - self.fraction_goals_rollout_goals)
            * self.fraction_resampled_goals_env_goals
        )
        num_replay_buffer_goals = int(
            batch_size * (1 - self.fraction_goals_rollout_goals)
            * self.fraction_resampled_goals_replay_buffer_goals
        )
        num_future_goals = batch_size - (num_rollout_goals + num_env_goals + num_replay_buffer_goals)

        new_obs_dict = self._batch_obs_dict(indices)
        new_next_obs_dict = self._batch_next_obs_dict(indices)

        if num_env_goals > 0:
            from inspect import signature
            sig = signature(self.env.sample_goals)
            if len(sig.parameters) == 2:
                keys = self.goal_keys + self.desired_goal_keys
                env_goals = self.env.sample_goals(num_env_goals, keys=set(keys))
            else:
                env_goals = self.env.sample_goals(num_env_goals)

            env_goals = preprocess_obs_dict(env_goals)
            last_env_goal_idx = num_rollout_goals + num_env_goals
            for desired_goal_key in self.desired_goal_keys:
                resampled_goals[desired_goal_key][num_rollout_goals:last_env_goal_idx] = (
                    env_goals[desired_goal_key]
                )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
                new_next_obs_dict[goal_key][num_rollout_goals:last_env_goal_idx] = \
                    env_goals[goal_key]
        if num_replay_buffer_goals > 0:
            replay_buffer_goals = self.sample_replay_buffer_states_as_goals(num_replay_buffer_goals)

            #replay_buffer_goals = preprocess_obs_dict(replay_buffer_goals) #<-------------------------------------------------
            last_env_goal_idx = num_rollout_goals + num_env_goals
            last_replay_buffer_goal_idx = num_rollout_goals + num_env_goals + num_replay_buffer_goals
            for desired_goal_key in self.desired_goal_keys:
                resampled_goals[desired_goal_key][last_env_goal_idx:last_replay_buffer_goal_idx] = (
                    replay_buffer_goals[desired_goal_key]
                )
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][last_env_goal_idx:last_replay_buffer_goal_idx] = \
                    replay_buffer_goals[goal_key]
                new_next_obs_dict[goal_key][last_env_goal_idx:last_replay_buffer_goal_idx] = \
                    replay_buffer_goals[goal_key]


        if num_future_goals > 0:
            future_obs_idxs = []
            for i in indices[-num_future_goals:]:
                possible_future_obs_idxs = self._idx_to_future_obs_idx[i]
                # This is generally faster than random.choice. Makes you wonder what
                # random.choice is doing
                num_options = len(possible_future_obs_idxs)
                next_obs_i = int(np.random.randint(0, num_options))
                future_obs_idxs.append(possible_future_obs_idxs[next_obs_i])
            future_obs_idxs = np.array(future_obs_idxs)
            for desired_goal_key in self.desired_goal_keys:
                resampled_goals[desired_goal_key][-num_future_goals:] = self._next_obs[
                    desired_goal_key.replace('desired', 'achieved')
                ][future_obs_idxs]
            for goal_key in self.goal_keys:
                new_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]
                new_next_obs_dict[goal_key][-num_future_goals:] = \
                    self._next_obs[goal_key][future_obs_idxs]

        # assign resampled_goals to the goals in new_obs_dict and new_next_obs_dict
        for desired_goal_key in self.desired_goal_keys:
            new_obs_dict[desired_goal_key] = resampled_goals[desired_goal_key]
            new_next_obs_dict[desired_goal_key] = resampled_goals[desired_goal_key]

        # post process new_obs_dict and new_next_obs_dict
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        new_next_obs_dict = postprocess_obs_dict(new_next_obs_dict)

        # compute rewards
        new_actions = self._actions[indices]
        new_rewards = self.env.compute_rewards(
            new_actions,
            new_next_obs_dict,
            # new_obs_dict,
        )
        if not self.vectorized:
            new_rewards = new_rewards.reshape(-1, 1)

        batch = {
            'observations': new_obs_dict[self.observation_key],
            'actions': new_actions,
            'rewards': new_rewards,
            'terminals': self._terminals[indices],
            'next_observations': new_next_obs_dict[self.observation_key],
            'resampled_goals': new_next_obs_dict[self.desired_goal_key],
            'indices': np.array(indices).reshape(-1, 1),
        }
        return batch

    def random_state_batch(self, batch_size):
        indices = self._sample_indices(batch_size)
        new_obs_dict = {self.observation_key: self._obs[self.observation_key][indices]}
        new_obs_dict = postprocess_obs_dict(new_obs_dict)
        return new_obs_dict[self.observation_key]

    def sample_replay_buffer_states_as_goals(self, batch_size):
        assert self._size > 0
        idxs = self._sample_indices(batch_size)
        goals = {}

        keys = self.goal_keys + self.desired_goal_keys

        for key in set(keys):
            if key in self.ob_spaces:
                goals[key] = self._next_obs[key.replace('desired', 'achieved')][idxs]

        return goals

    def _batch_obs_dict(self, indices):
        return {
            key: self._obs[key][indices]
            for key in self.ob_keys_to_save
        }

    def _batch_next_obs_dict(self, indices):
        return {
            key: self._next_obs[key][indices]
            for key in self.ob_keys_to_save
        }


def flatten_n(xs):
    xs = np.asarray(xs)
    return xs.reshape((xs.shape[0], -1))


def flatten_dict(dicts, keys):
    """
    Turns list of dicts into dict of np arrays
    """
    return {
        key: flatten_n([d[key] for d in dicts])
        for key in keys
    }

def preprocess_obs_dict(obs_dict):
    """
    Apply internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = unormalize_image(obs)
    return obs_dict

def postprocess_obs_dict(obs_dict):
    """
    Undo internal replay buffer representation changes: save images as bytes
    """
    for obs_key, obs in obs_dict.items():
        if 'image' in obs_key and obs is not None:
            obs_dict[obs_key] = normalize_image(obs)
    return obs_dict
