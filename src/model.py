import copy

import numpy as np
import torch

from agent import Agent
from config import *


def reset_coords(agent_location, other_agent_location):
    if agent_location[0] in range(other_agent_location[0] - 5, other_agent_location[0] + 5) and \
            agent_location[1] in range(other_agent_location[1] - 5, other_agent_location[1] + 5):
        return agent_location[0] - other_agent_location[0] + 5, agent_location[1] - other_agent_location[1] + 5
    return None


class Model:
    def __init__(self):
        """
        This should handle the logic of your model initialization like loading weights, setting constants, etc.
        """
        self.states, self.actions = None, None
        self.positions_xy = None
        self.position_hist = []
        self.mode = 'inference'
        assert self.mode in ['train', 'inference']
        self.path = './gdrive/MyDrive/trained_models/saved_agent_28052022_0933'

        self.model = Agent(buffer_size=BUFFER_SIZE, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, tau=TAU, gamma=GAMMA)

        if self.mode == 'inference':
            self.model.load(self.path)

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

    def update_state(self, state, position, target, other_states, other_agents):
        """
        Updates agents' states given the action of one agent
        """
        state = torch.tensor(state, device=DEVICE)
        action = self.model.get_action(state)

        updated_states = []
        if action:
            coords_update = self.update_coords(action, position)
            new_pos = (position[0] + coords_update[0], position[1] + coords_update[1])
            for other_state, agent in zip(other_states, other_agents):
                coords = reset_coords(position, agent)
                coords_ = reset_coords(new_pos, agent)
                if coords:
                    new_x, new_y = coords
                    other_state[1][new_x, new_y] = 0
                if coords_:
                    new_x, new_y = coords_
                    other_state[1][new_x, new_y] = 1
                updated_states.append(other_state)

            return action, updated_states, new_pos
        return action, other_states, position

    @staticmethod
    def manhattan(a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    def get_reward(self, position, new_position, target, done):
        hist = np.array(self.position_hist)
        hist = hist.reshape(hist.shape[1], hist.shape[2], -1)
        if done:
            return 100
        distance = self.manhattan(position, target)
        new_distance = self.manhattan(new_position, target)

        if new_position in hist:
            return -20
        if new_distance < distance:
            return 1
        if new_distance == distance:
            return -1
        return -2

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        """
        Given current observations, Done flags, agents' current positions and their targets, produce actions for agents.
        """
        # actions = []
        # updated_states = copy.deepcopy(obs)
        # for i in range(len(obs)):
        #     if not dones[i]:
        #         action, updated_states_tmp, new_position = self.update_state(updated_states[i], positions_xy[i],
        #                                                                      targets_xy[i], updated_states[i::],
        #                                                                      positions_xy[i::])
        #         updated_states = updated_states[0:i + 1] + updated_states_tmp
        #     else:
        #         action = 0
        #     actions.append(action)
        actions = self.model.get_action(obs)

        if self.mode == 'train':
            if self.states is None:
                self.states, self.actions = obs, actions
                self.positions_xy = positions_xy
                self.position_hist.append(np.array(positions_xy))
            else:
                rewards = np.array([self.get_reward(self.positions_xy[i], positions_xy[i], targets_xy[i], dones[i])
                                    for i in range(len(dones))])
                print('Rewards are:', rewards)
                for i in range(len(obs)):
                    self.model.add_to_replay_buffer(self.states[i], self.actions[i], rewards[i], obs[i], dones[i])
                self.model.train()
                self.states = obs

        return actions.detach().cpu().numpy().astype(int)

