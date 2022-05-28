import copy
import os
import subprocess
import time

import numpy as np

import torch
from torch.optim import AdamW

from replay_buffer import ReplayBuffer
from config import DEVICE, HIDDEN_SIZE


class Actor(torch.nn.Module):
    @staticmethod
    def update_coords(action, position):
        """
        Updates current agent's coordinates
        """
        x, y = position

        if action == 1:
            return x - 1, y
        if action == 2:
            return x + 1, y
        if action == 3:
            return x, y - 1
        if action == 4:
            return x, y + 1
        return x, y

    def update(self, probs, obstacles):
        action = torch.tensor(np.argmax(probs))
        x, y = self.update_coords(action, (5, 5))
        while obstacles[0][x, y] and action:
            probs[action] = 0
            action = torch.tensor(np.argmax(probs))
            x, y = self.update_coords(action, (5, 5))
        return action

    def __init__(self, name, hidden_size):
        super(Actor, self).__init__()
        self.net_name = name
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3).to(DEVICE).double()
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.AdaptiveMaxPool2d((16, 16)).to(DEVICE).double()
        self.flatten = torch.nn.Flatten(1)
        self.linear = torch.nn.Linear(hidden_size, 5).to(DEVICE).double()
        self.softmax = torch.nn.Softmax()
        self.trainable_layers = [self.conv, self.linear]

    def forward(self, state):
        state = state.double()
        conv_res = self.relu(self.conv(state))
        pool_res = self.pooling(conv_res)
        flatten_res = self.flatten(pool_res)
        probs = self.softmax(self.linear(flatten_res)).detach().cpu().numpy()
        state = state.detach().cpu()
        # print(probs.get_device())
        # print(state.get_device())
        res = np.zeros(len(state))
        for i, prob in enumerate(probs):
            res[i] = self.update(probs[i], state[i])

        return torch.tensor(res)

    def get_trainable_params(self):
        weights = []
        for layer in self.trainable_layers:
          weights.append(layer.weight)
        return weights


class Critic(torch.nn.Module):
    def __init__(self, name, hidden_size):
        super(Critic, self).__init__()
        self.net_name = name
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3).to(DEVICE)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.AdaptiveMaxPool2d((16, 16)).to(DEVICE)
        self.flatten = torch.nn.Flatten(1)
        self.linear = torch.nn.Linear(hidden_size + 1, 1).to(DEVICE)
        self.trainable_layers = [self.conv, self.linear]

    def forward(self, state, action): #добавить action
        action = action.to(DEVICE)
        conv_res = self.relu(self.conv(state))
        pool_res = self.pooling(conv_res)
        flatten_res = self.flatten(pool_res)
        flatten_res = torch.concat([flatten_res, action], dim=1).to(DEVICE).float()
        q_value = self.linear(flatten_res)
        return q_value

    def get_trainable_params(self):
        weights = []
        for layer in self.trainable_layers:
          weights.append(layer.weight)
        return weights


class Agent:
    def __init__(self, buffer_size, actor_lr, critic_lr, tau, gamma):
        self.replay_buffer = ReplayBuffer(
                        states_shape=(3, 11, 11), actions_shape=(1,), buffer_capacity=buffer_size
                    )

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau
        self.gamma = gamma

        self.actor = Actor('actor', HIDDEN_SIZE)
        self.critic = Critic('critic', HIDDEN_SIZE)

        self.target_actor = Actor('target_actor', HIDDEN_SIZE)
        self.target_critic = Critic('target_critic', HIDDEN_SIZE)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = AdamW(self.actor.get_trainable_params(), lr=self.actor_lr)
        self.critic_optimizer = AdamW(self.critic.get_trainable_params(), lr=self.critic_lr)

        self.critic_loss = torch.nn.MSELoss()

    def update_target_networks(self, tau: float):
        actor_weights = self.actor.state_dict()
        target_actor_weights = self.target_actor.state_dict()
        for key in target_actor_weights.keys():
            target_actor_weights[key] = tau * actor_weights[key] + (1 - tau) * target_actor_weights[key]

        self.target_actor.load_state_dict(target_actor_weights)

        critic_weights = self.critic.state_dict()
        target_critic_weights = self.target_critic.state_dict()

        for key in target_critic_weights.keys():
            target_critic_weights[key] = tau * critic_weights[key] + (1 - tau) * target_critic_weights[key]

        self.target_critic.load_state_dict(target_critic_weights)

    def add_to_replay_buffer(self, state, action, reward, new_state, done):
        return self.replay_buffer.add_record(state, action, reward, new_state, done)

    def save(self, path):
        date_now = time.strftime("%d%m%Y_%H%M")
        folder = path + f'/saved_agent_{date_now}'
        subprocess.run(['mkdir', folder])

        with open(f"{folder}/{self.actor.net_name}.pkl", 'wb') as file:
            torch.save(self.actor.state_dict(), file)

        with open(f"{folder}/{self.target_actor.net_name}.pkl", 'wb') as file:
            torch.save(self.target_actor.state_dict(), file)

        with open(f"{folder}/{self.critic.net_name}.pkl", 'wb') as file:
            torch.save(self.critic.state_dict(), file)

        with open(f"{folder}/{self.target_critic.net_name}.pkl", 'wb') as file:
            torch.save(self.target_critic.state_dict(), file)

        self.replay_buffer.save(folder)


    def load(self, path):
        self.actor.load_state_dict(torch.load(f'{path}/{self.actor.net_name}.pkl'))
        self.target_actor.load_state_dict(torch.load(f'{path}/{self.target_actor.net_name}.pkl'))
        self.critic.load_state_dict(torch.load(f'{path}/{self.critic.net_name}.pkl'))
        self.target_critic.load_state_dict(torch.load(f'{path}/{self.target_critic.net_name}.pkl'))

        # self.replay_buffer.load(f"{path}")


    def get_action(self, state):
        state = torch.tensor(state, device=DEVICE)
        return self.actor(state)

    def train(self):
        if self.replay_buffer.check_buffer_size() == False:
            return

        state, action, reward, new_state, done = self.replay_buffer.get_minibatch()

        states = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        new_states = torch.tensor(new_state, dtype=torch.float32, device=DEVICE)
        rewards = torch.tensor(reward, dtype=torch.float32, device=DEVICE)
        actions = torch.tensor(action, dtype=torch.float32, device=DEVICE)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        target_actions = torch.unsqueeze(self.target_actor(new_states), 1)
        target_critic_values = torch.squeeze(
            self.target_critic(new_states, target_actions), 1
        ).detach().cpu()
        critic_value = torch.squeeze(self.critic(states, actions), 1)
        target = rewards + torch.tensor(self.gamma * target_critic_values * (1 - done),
                                        dtype=torch.float32, device=DEVICE)
        critic_loss = self.critic_loss(critic_value, target)

        critic_loss.backward()
        self.critic_optimizer.step()

        policy_actions = torch.unsqueeze(self.actor(states), 1)
        actor_loss = -self.critic(states, policy_actions)
        actor_loss = torch.mean(actor_loss)

        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks(self.tau)