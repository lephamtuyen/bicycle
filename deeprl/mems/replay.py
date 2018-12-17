from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from mems.SumTree import *
import random

class Memory(object):
    """
    An implementation of the replay memory. This is essential when dealing with DRL algorithms that are not
    multi-threaded as in A3C.
    """

    def __init__(self, memory_size, state_dim, action_dim, batch_size):
        """
        Args
            memory_size: size of memory replay
            state_dim: state dimension
            action_dim: action dimension
            batch_size: number of samples which is sampled for training
        A naive implementation of the replay memory, need to do more work on this after testing DDPG
        """
        self.memory_size = memory_size
        self.batch_size = batch_size

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.curr_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # next state
        self.next_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # reward
        self.rewards = np.empty(memory_size, dtype=np.float32)
        # terminal
        self.terminals = np.empty(memory_size, dtype=np.float32)
        # actions
        self.actions = np.empty((memory_size, action_dim) if action_dim > 1 else memory_size, dtype=np.float32)

        self.current = 0
        self.count = 0

    def add(self, sample, error=None):
        self.curr_state[self.current, ...] = sample[0]
        self.next_state[self.current, ...] = sample[1]
        self.rewards[self.current] = sample[2]
        self.terminals[self.current] = sample[3]
        self.actions[self.current] = sample[4]

        self.current += 1
        self.count = max(self.count, self.current)
        if self.current >= self.memory_size - 1:
            self.current = 0

    def sample(self):
        indexes = np.random.randint(0, self.count, self.batch_size)

        curr_state = self.curr_state[indexes, ...]
        next_state = self.next_state[indexes, ...]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        actions = self.actions[indexes]
        return [curr_state, next_state, rewards, terminals, actions], None

    def save(self, save_dir):
        path = os.path.join(save_dir, type(self).__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving memory...")
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            np.save(os.path.join(path, name), arr=getattr(self, name))

    def update(self, idxs=None, errors=None):
        return

    def get_curr_size(self):
        return self.count

    def restore(self, save_dir):
        """
        Restore the memory.
        """
        path = os.path.join(save_dir, type(self).__name__)
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            setattr(self, name, np.load(os.path.join(path, "%s.npy" % name)))

    def size(self):
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            print("%s size is %s" % (name, getattr(self, name).shape))

class PrioritizedMemory(object):
    e = 0.01
    a = 0.6

    def __init__(self, memory_size, state_dim, action_dim, batch_size):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tree = SumTree(memory_size)

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.sample_curr_state = np.empty(shape=(batch_size, ) + state_dim, dtype=np.float32)
        # next state
        self.sample_next_state = np.empty(shape=(batch_size, ) + state_dim, dtype=np.float32)
        # reward
        self.sample_rewards = np.empty(batch_size, dtype=np.float32)
        # terminal
        self.sample_terminals = np.empty(batch_size, dtype=np.float32)
        # actions
        self.sample_actions = np.empty((batch_size, action_dim) if action_dim > 1 else batch_size, dtype=np.float32)

    def get_curr_size(self):
        return self.tree.total()

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, sample, error=None):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self):
        idxs = []
        segment = self.tree.total() / self.batch_size

        # min_bound = segment * numpy.arange(self.batch_size)
        # s = min_bound + segment * np.random.rand(self.batch_size)
        # s = s.astype(int)
        # (idxs, p, data) = self.tree.get_batch(s)


        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)

            idxs.append(idx)

            # current state
            self.sample_curr_state[i] = data[0]
            # next state
            self.sample_next_state[i] = data[1]
            # reward
            self.sample_rewards[i] = data[2]
            # terminal
            self.sample_terminals[i] = data[3]
            # actions
            self.sample_actions[i] = data[4]

        return [self.sample_curr_state, self.sample_next_state, self.sample_rewards, self.sample_terminals, self.sample_actions], idxs

    def update(self, idxs=None, errors=None):
        for i in range(self.batch_size):
            p = self._getPriority(errors[i])
            self.tree.update(idxs[i], p)

    def save(self, save_dir):
        path = os.path.join(save_dir, type(self).__name__)
        if not os.path.exists(path):
            os.makedirs(path)
        print("Saving memory...")
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            np.save(os.path.join(path, name), arr=getattr(self, name))

    def restore(self, save_dir):
        """
        Restore the memory.
        """
        path = os.path.join(save_dir, type(self).__name__)
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            setattr(self, name, np.load(os.path.join(path, "%s.npy" % name)))

    def size(self):
        for name in ("curr_state", "next_state", "rewards", "terminals", "actions"):
            print("%s size is %s" % (name, getattr(self, name).shape))

class ActorMemory(object):
    """
    An implementation of the replay memory. This is essential when dealing with DRL algorithms that are not
    multi-threaded as in A3C.
    """

    def __init__(self, memory_size, state_dim, action_dim, batch_size):
        """
        A naive implementation of the replay memory, need to do more work on this after testing DDPG
        """
        self.memory_size = memory_size
        self.batch_size = batch_size

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.curr_state = np.empty(shape=(memory_size, ) + state_dim)
        # goal
        self.goal = np.empty(shape=(memory_size,) + state_dim)
        # next state
        self.next_state = np.empty(shape=(memory_size, ) + state_dim)
        # reward
        self.rewards = np.empty(memory_size)
        # actions
        self.actions = np.empty((memory_size, action_dim) if action_dim > 1 else memory_size)

        self.current = 0
        self.count = 0

    def add(self, curr_state, goal, next_state, reward, action):
        self.curr_state[self.current, ...] = curr_state
        self.goal[self.current, ...] = goal
        self.next_state[self.current, ...] = next_state
        self.rewards[self.current] = reward
        self.actions[self.current] = action

        self.current += 1
        self.count = max(self.count, self.current)
        if self.current >= self.memory_size - 1:
            self.current = 0

    def sample(self):
        indexes = np.random.randint(0, self.count, self.batch_size)

        curr_state = self.curr_state[indexes, ...]
        goal = self.goal[indexes, ...]
        next_state = self.next_state[indexes, ...]
        rewards = self.rewards[indexes]
        actions = self.actions[indexes]
        return curr_state, goal, next_state, rewards, actions

class MetaMemory(object):
    """
    An implementation of the replay memory. This is essential when dealing with DRL algorithms that are not
    multi-threaded as in A3C.
    """

    def __init__(self, memory_size, state_dim, goal_dim, batch_size):
        """
        A naive implementation of the replay memory, need to do more work on this after testing DDPG
        """
        self.memory_size = memory_size
        self.batch_size = batch_size

        if type(state_dim) is not tuple:
            state_dim = (state_dim, )

        # current state
        self.curr_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # goal
        self.goal = np.empty((memory_size, goal_dim) if goal_dim > 1 else memory_size, dtype=np.float32)
        # next state
        self.next_state = np.empty(shape=(memory_size, ) + state_dim, dtype=np.float32)
        # reward
        self.rewards = np.empty(memory_size, dtype=np.float32)

        self.current = 0
        self.count = 0

    def add(self, curr_state, goal, next_state, reward):
        self.curr_state[self.current, ...] = curr_state
        self.goal[self.current, ...] = goal
        self.next_state[self.current, ...] = next_state
        self.rewards[self.current] = reward

        self.current += 1
        self.count = max(self.count, self.current)
        if self.current >= self.memory_size - 1:
            self.current = 0

    def sample(self):
        indexes = np.random.randint(0, self.count, self.batch_size)

        curr_state = self.curr_state[indexes, ...]
        goal = self.goal[indexes, ...]
        next_state = self.next_state[indexes, ...]
        rewards = self.rewards[indexes]
        return curr_state, goal, next_state, rewards



