import os
import json
from typing import Union

import numpy as np


BUFFER_CAPACITY = 10000
BATCH_SIZE = 4
MIN_SIZE_BUFFER = 10


class ReplayBuffer:
    """
    The replay buffer class used to store played sessions.
    """

    def __init__(
        self,
        states_shape: Union[tuple, list],
        actions_shape: Union[tuple, list],
        buffer_capacity: int = BUFFER_CAPACITY,
        batch_size: int = BATCH_SIZE,
        min_size_buffer: int = MIN_SIZE_BUFFER,
    ):
        """
        Constructor method.

        :param states_shape: Shape of the states used in environtment.
        :type states_shape: Union[list, tuple]
        :param actions_shape: Shape of the actions used in environment.
        :type actions_shape: Union[list, tuple]
        :param buffer_capacity: Size of the buffer, defaults to BUFFER_CAPACITY=10000.
        :type buffer_capacity: int, optional
        :param batch_size: Size of the returned batch, defaults to BATCH_SIZE=4.
        :type batch_size: int, optional
        :param min_size_buffer: Minimum buffer size before starting using it, that is returning mini-batches, defaults to MIN_SIZE_BUFFER=10
        :type min_size_buffer: int, optional
        """
        self.states_shape = states_shape
        self.actions_shape = actions_shape
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.min_size_buffer = min_size_buffer
        self.buffer_counter = 0
        self.n_games = 0

        self.states = np.zeros((self.buffer_capacity, *self.states_shape))
        self.actions = np.zeros((self.buffer_capacity, *self.actions_shape))
        self.rewards = np.zeros((self.buffer_capacity))
        self.next_states = np.zeros((self.buffer_capacity, *self.states_shape))
        self.dones = np.zeros((self.buffer_capacity), dtype=bool)

    def __len__(self):
        """
        Method for returning length of the buffer.

        :return: Number of written episodes.
        :rtype: int
        """
        return self.buffer_counter

    def add_record(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: Union[int, float],
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add episode record to the buffer. If the buffer size is more than buffer capasity
        the oldest records are overridden.

        :param state: Environment state.
        :type state: np.ndarray
        :param state_rgb: Image from RGB camera
        :type state_rgb: np.ndarray
        :param action: Action taken.
        :type action: np.ndarray
        :param reward: Collected reward.
        :type reward: Union[int, float]
        :param next_state: Produced environment state.
        :type next_state: np.ndarray
        :param done: A flag for terminal state.
        :type done: bool
        :param coordinate: Vehicle location
        :type coordinate: np.ndarray
        """
        index = self.buffer_counter % self.buffer_capacity

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = done

        # Update the counter when record something
        self.buffer_counter += 1

    def check_buffer_size(self):
        """
        Check replay buffer if it has minimum number of episodes and can return batches.

        :return: ``True`` if ``buffer_counter`` is more than ``batch_size`` and ``min_size_buffer``,
            ``False`` otherwise.
        :rtype: bool
        """
        return (
            self.buffer_counter >= self.batch_size
            and self.buffer_counter >= self.min_size_buffer
        )

    def update_n_games(self):
        """
        Update games counter.
        """
        self.n_games += 1

    def get_minibatch(self):
        """
        Get the minibatch of specified size.

        :return: Returns minibatch in ``(state, action, reward, next_state, done)`` format.
        :rtype: tuple
        """
        # If the counter is less than the capacity we don't want to take zeros records,
        # if the cunter is higher we don't access the record using the counter
        # because older records are deleted to make space for new one
        buffer_range = min(self.buffer_counter, self.buffer_capacity)

        batch_index = np.random.choice(buffer_range, self.batch_size, replace=False)

        # Take indices
        state = self.states[batch_index]
        action = self.actions[batch_index]
        reward = self.rewards[batch_index]
        next_state = self.next_states[batch_index]
        done = self.dones[batch_index]

        return state, action, reward, next_state, done

    def save(self, folder_name: str):
        """
        Save the replay buffer.

        :param folder_name: Name or path to the folder to save replay buffer.
        :type folder_name: str
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        np.save(os.path.join(folder_name, "states.npy"), self.states)
        np.save(os.path.join(folder_name, "states_rgb.npy"), self.states)
        np.save(os.path.join(folder_name, "actions.npy"), self.actions)
        np.save(os.path.join(folder_name, "rewards.npy"), self.rewards)
        np.save(os.path.join(folder_name, "next_states.npy"), self.next_states)
        np.save(os.path.join(folder_name, "dones.npy"), self.dones)
        np.save(os.path.join(folder_name, "coordinates.npy"), self.coordinates)

        dict_info = {"buffer_counter": self.buffer_counter, "n_games": self.n_games}
        with open(os.path.join(folder_name, "dict_info.json"), "w") as f:
            json.dump(dict_info, f)

    def load(self, folder_name: str):
        """
        Load the replay buffer and all its components.

        :param folder_name: Name or path to the folder with saved replay buffer.
        :type folder_name: str
        """
        self.states = np.load(os.path.join(folder_name, "states.npy"))
        self.states_rgb = np.load(os.path.join(folder_name, "states_rgb.npy"))
        self.actions = np.load(os.path.join(folder_name, "actions.npy"))
        self.rewards = np.load(os.path.join(folder_name, "rewards.npy"))
        self.next_states = np.load(os.path.join(folder_name, "next_states.npy"))
        self.dones = np.load(os.path.join(folder_name, "dones.npy"))
        self.coordinates = np.load(os.path.join(folder_name, "coordinates.npy"))

        with open(os.path.join(folder_name, "dict_info.json"), "r") as f:
            dict_info = json.load(f)

        self.buffer_counter = dict_info["buffer_counter"]
        self.n_games = dict_info["n_games"]
