
import os
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import load_episodes

class SemIRLDataset(Dataset):
    def __init__(self, directory):
        self.episodes = self._flatten(load_episodes(directory))
        self.episodes['grid_count'] = self.episodes['grid_count'].permute(0, 3, 1, 2)

    def _flatten(self, episodes):
        """
        Flatten the episode and timestep dimensions
        """
        obs_keys = ['grid_count', 'agent_pos', 'goal_pos', 'action']
        flatten_episodes = {k: [] for k in obs_keys}
        for name, episode in episodes.items():
            for k in obs_keys:
                flatten_episodes[k].append(episode[k])

        for k, v in flatten_episodes.items():
            flatten_episodes[k] = torch.from_numpy(np.concatenate(v, axis=0))
        return flatten_episodes

    def __len__(self):
        return len(self.episodes['action'])

    def __getitem__(self, idx):
        grid_count = self.episodes['grid_count'][idx]
        agent_pos = self.episodes['agent_pos'][idx]
        goal_pos = self.episodes['goal_pos'][idx]
        action = self.episodes['action'][idx]
        return grid_count, agent_pos, goal_pos, action