from source.model import ActorCritic
from source.carlaenv import CarlaEnv
import source.utility as util
from source.replaybuffer import ReplayBuffer
from source.model import device
from tqdm import tqdm
import torch


class Trainer(object):
    def __init__(self):
        self.env = CarlaEnv()
        self.config = util.get_env_settings("./config.yaml")
        self.ac_net = ActorCritic(
            4, self.config['hidden_dim'], self.config['n_layers'], self.config['gamma'], self.config['lr']).to(device)
        self.replaybuffer = ReplayBuffer(self.config['buffer_size'])

    def train(self, epoch_i):
        """on-policy actor-critic training."""
        # sample n trajectories
        print("sample trajectories")
        paths = util.sample_n_trajectories(
            self.config['sample_n'], self.env, self.ac_net, self.config['max_episode_length'], epoch_i)
        # add trajectories to replaybuffer
        print("add to replaybuffer")
        self.replaybuffer.add_rollouts(paths)
        # sample lastest trajectories for training
        print("update policy")
        training_paths = self.replaybuffer.sample_recent_rollouts(
            self.config['training_n'])
        self.ac_net.update(training_paths, epoch_i)

    def training_loop(self):
        print("*" * 20)
        print(f"use device {device}")
        print("*" * 20)
        for i in tqdm(range(self.config['epoch']), desc="Epoch"):
            self.train(i)
            if (i + 1) % 10 == 0:
                print(f"save checkpoint at {i + 1}")
                torch.save(self.ac_net.state_dict(),
                           f"./checkpoints/model{i}.pt")
