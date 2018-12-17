from gym import Wrapper
from gym.wrappers.time_limit import *
from utils.utilities import rgb2grey
import numpy as np
from gym.spaces import *
import gym

class ContinuousWrapper(Wrapper):
    def __init__(self, env, is_hierarchical=False):
        """
        Create a wrapper for OpenAI Gym environment. This wrapper normalizes the game state between
        min_value and max_value
        """
        super(ContinuousWrapper, self).__init__(env)

        if type(env) == TimeLimit:
            self.env = env.env

        act_space = self.env.action_space
        obs_space = self.env.observation_space

        if not type(act_space) == gym.spaces.box.Box:
            raise RuntimeError('Environment with continous action space (i.e. Box) required.')
        if not type(obs_space) == gym.spaces.box.Box:
            raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

        # Observation space
        self.obs_k = np.ones(obs_space.shape)
        self.obs_b = np.zeros(obs_space.shape)
        for (idx, value) in  np.ndenumerate(obs_space.high):
            if (obs_space.high[idx] < 1e10 and obs_space.low[idx] > -1e10):
                self.obs_k[idx] = (obs_space.high[idx] - obs_space.low[idx]) / 2.
                self.obs_b[idx] = (obs_space.high[idx] + obs_space.low[idx]) / 2.

        # if np.any(obs_space.high < 1e10) and np.any(obs_space.low > -1e10):
        #     h = obs_space.high
        #     l = obs_space.low
        #     self.obs_k = (h - l) / 2.
        #     self.obs_b = (h + l) / 2.
        # else:
        #     self.obs_k = np.ones_like(obs_space.high)
        #     self.obs_b = np.zeros_like(obs_space.high)

        # Action space
        h = act_space.high
        l = act_space.low
        self.act_k = (h - l) / 2.
        self.act_b = (h + l) / 2.

        # Check and assign transformed spaces
        self.env.observation_space = gym.spaces.Box(self.__normalize_state(obs_space.low),
                                                    self.__normalize_state(obs_space.high))

        self.is_hierarchical = is_hierarchical
        if (self.is_hierarchical == True):
            subgoal_space = self.env.subgoal_space
            h = subgoal_space.high
            l = subgoal_space.low
            self.subgoal_k = (h - l) / 2.
            self.subgoal_b = (h + l) / 2.

    def __normalize_state(self, obs):
        return (obs - self.obs_b) / self.obs_k

    def __normalize_action(self, action):
        return (action - self.act_b) / self.act_k

    def __normalize_subgoal(self, subgoal):
        return (subgoal - self.subgoal_b) / self.subgoal_k

    def __denormalize_action(self, action):
        return np.clip(self.act_k * action + self.act_b, self.env.action_space.low, self.env.action_space.high)

    def __denormalize_subgoal(self, subgoal):
        return np.clip(self.subgoal_k * subgoal + self.subgoal_b, self.env.subgoal_space.low, self.env.subgoal_space.high)

    def _step(self, action):
        ac_f = self.__denormalize_action(action)
        obs, reward, term, info = self.env.step(ac_f)
        obs_f = self.__normalize_state(obs)
        return obs_f, reward, term, info

    def _reset(self):
        obs = self.env.reset()
        return self.__normalize_state(obs)

    def sample_action(self):
        action = self.env.action_space.sample()
        return self.__normalize_action(action)

    def random_subgoal(self):
        subgoal, position = self.env.random_subgoal()
        return self.__normalize_subgoal(subgoal), position

    def setSubgoal(self,subgoal):
        subgoal_f = self.__denormalize_subgoal(subgoal)
        position = self.env._setSubgoal(subgoal_f)
        return position



class DiscreteWrapper(Wrapper):
    def __init__(self, env):
        """
        Create a wrapper for OpenAI Gym environment. This wrapper normalizes the game state between
        min_value and max_value
        """
        super(DiscreteWrapper, self).__init__(env)

        act_space = self.env.action_space
        obs_space = self.env.observation_space

        if not type(act_space) == gym.spaces.Discrete:
            raise RuntimeError('Environment with discrete action space (i.e. Discrete) required.')
        if not type(obs_space) == gym.spaces.box.Box:
            raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

        # Observation space
        self.obs_k = np.ones(obs_space.shape)
        self.obs_b = np.zeros(obs_space.shape)
        for (idx, value) in  np.ndenumerate(obs_space.high):
            if (obs_space.high[idx] < 1e10 and obs_space.low[idx] > -1e10):
                self.obs_k[idx] = (obs_space.high[idx] - obs_space.low[idx]) / 2.
                self.obs_b[idx] = (obs_space.high[idx] + obs_space.low[idx]) / 2.

        # Action space
        h = act_space.high
        l = act_space.low
        self.act_k = (h - l) / 2.
        self.act_b = (h + l) / 2.

        # Check and assign transformed spaces
        self.env.observation_space = gym.spaces.Box(self.__normalize_state(obs_space.low),
                                                    self.__normalize_state(obs_space.high))

    def __normalize_state(self, obs):
        return (obs - self.obs_b) / self.obs_k

    def __normalize_action(self, action):
        return (action - self.act_b) / self.act_k

    def __denormalize_action(self, action):
        return np.clip(self.act_k * action + self.act_b, self.env.action_space.low, self.env.action_space.high)

    def _step(self, action):
        ac_f = self.__denormalize_action(action)
        obs, reward, term, info = self.env.step(ac_f)
        obs_f = self.__normalize_state(obs)
        return obs_f, reward, term, info

    def _reset(self):
        obs = self.env.reset()
        return self.__normalize_state(obs)

    def sample_action(self):
        action = self.env.action_space.sample()
        return self.__normalize_action(action)