# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image dataset."""

import os
import pickle

import numpy as np
from ravens import tasks
from ravens.tasks import cameras
import tensorflow as tf

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]


class Dataset:
  """A simple image dataset class."""

  def __init__(self, path):
    """A simple RGB-D image dataset."""
    self.path = path
    self.sample_set = []
    self.max_seed = -1
    self.n_episodes = 0

    # Track existing dataset if it exists.
    color_path = os.path.join(self.path, 'action')
    if tf.io.gfile.exists(color_path):
      for fname in sorted(tf.io.gfile.listdir(color_path)):
        if '.pkl' in fname:
          seed = int(fname[(fname.find('-') + 1):-4])
          self.n_episodes += 1
          self.max_seed = max(self.max_seed, seed)

    self._cache = {}

  def interpolate_poses(self, start_pose, end_pose, num_steps):
      # Unpack the start and end poses
      start_pos, start_quat = start_pose
      end_pos, end_quat = end_pose

      # Create position interpolation
      positions = np.linspace(start_pos, end_pos, num=num_steps + 1)

      # Combine positions and quaternions
      interpolated_actions = []
      for i in range(num_steps):
          action = {
              'pose0': ((positions[i]), [0, 0, 0 , 1]),
              'pose1': (positions[i+1], [0, 0, 0 , 1])
          }
          interpolated_actions.append(action)

      return interpolated_actions

  def add(self, seed, episode):
      """Add an episode to the dataset with sequences of observations.

      Args:
        seed: random seed used to initialize the episode.
        episode: list of (obs_sequence, act, reward, info) tuples.
      """
      color, depth, action, reward, info, segm = [], [], [], [], [], []
      for obs, act, r, i in episode:
          # num_obs = len(obs_sequence)
          # if num_obs > 1 and act is not None:
          #     # Interpolate poses from act['pose0'] to act['pose1']
          #     interpolated_actions = self.interpolate_poses(act['pose0'], act['pose1'], num_obs)
          # else:
          #     interpolated_actions = [act]  # No interpolation needed if only one obs

          # for idx, obs in enumerate(obs_sequence):
          #     color.append(np.uint8(obs['color']))
          #     depth.append(obs['depth'])
          #     segm.append(np.uint8(obs['segm']))
          #     info.append(i)
          # # print("Interpolated Actions: ", interpolated_actions)
          # if interpolated_actions is not None:
          #   for interpolated_action in interpolated_actions:
          #     # print("Appending interpolation: ", interpolated_action)
          #     action.append(interpolated_action)
          # else:
          #   print("Appending: ", act)
          #   action.append(act)
          # # action.append({'pose0': interpolated_actions[idx][0], 'pose1': interpolated_actions[idx][1]})
          color.append(np.uint8(obs['color']))
          depth.append(obs['depth'])
          segm.append(np.uint8(obs['segm']))
          info.append(i)
          action.append(act)
          reward.append(r)
          # reward.extend([r / num_obs] * num_obs)  # Distribute reward evenly across observations

      # Processing and saving the accumulated data
      color = np.array(color, dtype=np.uint8)
      depth = np.array(depth, dtype=np.float32)
      segm = np.array(segm, dtype=np.uint8)

      def dump(data, field):
          field_path = os.path.join(self.path, field)
          if not tf.io.gfile.exists(field_path):
              tf.io.gfile.makedirs(field_path)
          fname = f'{self.n_episodes:06d}-{seed}.pkl'
          with tf.io.gfile.GFile(os.path.join(field_path, fname), 'wb') as f:
              pickle.dump(data, f)

      dump(color, 'color')
      dump(depth, 'depth')
      dump(action, 'action')
      dump(reward, 'reward')
      dump(info, 'info')
      dump(segm, 'segm')

      self.n_episodes += 1
      self.max_seed = max(self.max_seed, seed)


  def set(self, episodes):
    """Limit random samples to specific fixed set."""
    self.sample_set = episodes

  def load(self, episode_id, images=True, cache=False):
    """Load data from a saved episode.

    Args:
      episode_id: the ID of the episode to be loaded.
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      episode: list of (obs, act, reward, info) tuples.
      seed: random seed used to initialize the episode.
    """

    def load_field(episode_id, field, fname):

      # Check if sample is in cache.
      if cache:
        if episode_id in self._cache:
          if field in self._cache[episode_id]:
            return self._cache[episode_id][field]
        else:
          self._cache[episode_id] = {}

      # Load sample from files.
      path = os.path.join(self.path, field)
      data = pickle.load(open(os.path.join(path, fname), 'rb'))
      if cache:
        self._cache[episode_id][field] = data
      return data

    # Get filename and random seed used to initialize episode.
    seed = None
    path = os.path.join(self.path, 'action')
    for fname in sorted(tf.io.gfile.listdir(path)):
      if f'{episode_id:06d}' in fname:
        seed = int(fname[(fname.find('-') + 1):-4])

        # Load data.
        color = load_field(episode_id, 'color', fname)
        depth = load_field(episode_id, 'depth', fname)
        action = load_field(episode_id, 'action', fname)
        reward = load_field(episode_id, 'reward', fname)
        info = load_field(episode_id, 'info', fname)

        # Reconstruct episode.
        episode = []
        for i in range(len(action)):
          obs = {'color': color[i], 'depth': depth[i]} if images else {}
          episode.append((obs, action[i], reward[i], info[i]))
        return episode, seed

  def sample(self, images=True, cache=False):
    """Uniformly sample from the dataset.

    Args:
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      sample: randomly sampled (obs, act, reward, info) tuple.
      goal: the last (obs, act, reward, info) tuple in the episode.
    """

    # Choose random episode.
    if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
      episode_id = np.random.choice(self.sample_set)
    else:
      episode_id = np.random.choice(range(self.n_episodes))
    episode, _ = self.load(episode_id, images, cache)

    # Return random observation action pair (and goal) from episode.
    i = np.random.choice(range(len(episode) - 1))
    sample, goal = episode[i], episode[-1]
    return sample, goal
