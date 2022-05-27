import numpy as np

import torch
from torch.optim import AdamW

from replay_buffer import ReplayBuffer
from config import DEVICE, HIDDEN_SIZE


class Actor(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Actor, self).__init__()
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
        probs = self.softmax(self.linear(flatten_res))
        return np.argmax(probs.detach().cpu().numpy())

    def get_trainable_params(self):
        weights = []
        for layer in self.trainable_layers:
          weights.append(layer.weight)
        return weights


class Critic(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Critic, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3).to(DEVICE)
        self.relu = torch.nn.ReLU()
        self.pooling = torch.nn.AdaptiveMaxPool2d((16, 16)).to(DEVICE)
        self.flatten = torch.nn.Flatten(1)
        self.linear = torch.nn.Linear(hidden_size + 1, 1).to(DEVICE)
        self.trainable_layers = [self.conv, self.linear]

    def forward(self, state, action): #добавить action
        action = torch.tensor(action.reshape(action.shape[1], -1))
        conv_res = self.relu(self.conv(state))
        pool_res = self.pooling(conv_res)
        flatten_res = self.flatten(pool_res)
        flatten_res = torch.concat([flatten_res, action], dim=1)
        q_value = self.linear(flatten_res)
        return q_value

    def get_trainable_params(self):
        weights = []
        for layer in self.trainable_layers:
          weights.append(layer.weight)
        return weights


class Agent:
    def __init__(self, buffer_size, actor_lr, critic_lr, tau):
        self.replay_buffer = ReplayBuffer(
                        states_shape=(3, 11, 11), actions_shape=(1,), buffer_capacity=buffer_size
                    )

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau = tau

        self.actor = Actor(HIDDEN_SIZE)
        self.critic = Critic(HIDDEN_SIZE)

        self.target_actor = Actor(HIDDEN_SIZE)
        self.target_critic = Critic(HIDDEN_SIZE)

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
        pass

    def load(self, path):
        pass

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

        target_actions = self.target_actor(new_states)
        target_critic_values = torch.squeeze(
            self.target_critic(new_states, target_actions), 1
        ).detach().cpu()
        critic_value = torch.squeeze(self.critic(states, actions), 1)
        target = rewards + torch.tensor(self.gamma * target_critic_values * (1 - done),
                                        dtype=torch.float32, device=DEVICE)
        critic_loss = self.critic_loss(critic_value, target)

        critic_loss.backward()
        self.critic_optimizer.step()

        policy_actions = self.actor(states)
        actor_loss = -self.critic(states, policy_actions)
        actor_loss = torch.mean(actor_loss)

        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks_yolo(self.tau)