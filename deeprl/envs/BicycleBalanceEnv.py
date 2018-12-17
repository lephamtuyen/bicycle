import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign
from matplotlib import pyplot as plt

class BicycleBalanceEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # Environment parameters.
        self.time_step = 0.01

        # Goal position and radius
        # Lagouakis (2002) uses angle to goal, not heading, as a state
        self.r_goal = 5.
        self.max_distance = 1000.

        # Acceleration on Earth's surface due to gravity (m/s^2):
        self.g = 9.82

        # See the paper for a description of these quantities:
        # Distances (in meters):
        self.c = 0.66
        self.dCM = 0.30
        self.h = 0.94
        self.L = 1.11
        self.r = 0.34
        # Masses (in kilograms):
        self.Mc = 15.0
        self.Md = 1.7
        self.Mp = 60.0
        # Velocity of a bicycle (in meters per second), equal to 10 km/h:
        self.v = 10.0 * 1000.0 / 3600.0

        # Derived constants.
        self.M = self.Mc + self.Mp  # See Randlov's code.
        self.Idc = self.Md * self.r ** 2
        self.Idv = 1.5 * self.Md * self.r ** 2
        self.Idl = 0.5 * self.Md * self.r ** 2
        self.Itot = 13.0 / 3.0 * self.Mc * self.h ** 2 + self.Mp * (self.h + self.dCM) ** 2
        self.sigmad = self.v / self.r

        # Angle at which to fail the episode
        self.omega_threshold = np.pi / 6
        self.theta_threshold = np.pi/2
        self.max_torque = 2.0

        high = np.array([
            self.theta_threshold,
            np.finfo(np.float32).max,
            self.omega_threshold,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.reset()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        T = np.clip(u, -self.max_torque, self.max_torque)[0]
        d = 0.0

        # For recordkeeping.
        # ------------------
        self.xfhist.append(self.xf)
        self.yfhist.append(self.yf)
        self.xbhist.append(self.xb)
        self.ybhist.append(self.yb)

        # Intermediate time-dependent quantities.
        # ---------------------------------------
        # Avoid divide-by-zero, just as Randlov did.
        if self.theta == 0:
            rf = 1e8
            rb = 1e8
            rCM = 1e8
        else:
            rf = self.L / np.abs(sin(self.theta))
            rb = self.L / np.abs(tan(self.theta))
            rCM = sqrt((self.L - self.c) ** 2 + self.L ** 2 / tan(self.theta) ** 2)

        phi = self.omega + np.arctan(d / self.h)

        # Equations of motion.
        # --------------------
        # Second derivative of angular acceleration:
        self.omegadd = 1 / self.Itot * (self.M * self.h * self.g * sin(phi)
                                   - cos(phi) * (self.Idc * self.sigmad * self.thetad
                                                 + sign(self.theta) * self.v ** 2 * (
                                                     self.Md * self.r * (1.0 / rf + 1.0 / rb)
                                                     + self.M * self.h / rCM)))
        self.thetadd = (T - self.Idv * self.sigmad * self.omegad) / self.Idl

        # Integrate equations of motion using Euler's method.
        # ---------------------------------------------------
        # yt+1 = yt + yd * dt.
        # Must update omega based on PREVIOUS value of omegad.
        self.omegad += self.omegadd * self.time_step
        self.omega += self.omegad * self.time_step
        self.thetad += self.thetadd * self.time_step
        self.theta += self.thetad * self.time_step

        # Handlebars can't be turned more than 80 degrees.
        self.theta = np.clip(self.theta, -1.3963, 1.3963)

        # Wheel ('tyre') contact positions.
        # ---------------------------------

        # Front wheel contact position.
        front_temp = self.v * self.time_step / (2 * rf)
        # See Randlov's code.
        if front_temp > 1:
            front_temp = sign(self.psi + self.theta) * 0.5 * np.pi
        else:
            front_temp = sign(self.psi + self.theta) * arcsin(front_temp)

        self.xf += self.v * self.time_step * -sin(self.psi + self.theta + front_temp)
        self.yf += self.v * self.time_step * cos(self.psi + self.theta + front_temp)

        # Rear wheel.
        back_temp = self.v * self.time_step / (2 * rb)
        # See Randlov's code.
        if back_temp > 1:
            back_temp = np.sign(self.psi) * 0.5 * np.pi
        else:
            back_temp = np.sign(self.psi) * np.arcsin(back_temp)

        self.xb += self.v * self.time_step * -sin(self.psi + back_temp)
        self.yb += self.v * self.time_step * cos(self.psi + back_temp)

        # Preventing numerical drift.
        # ---------------------------
        # Copying what Randlov did.
        current_wheelbase = sqrt((self.xf - self.xb) ** 2 + (self.yf - self.yb) ** 2)
        if np.abs(current_wheelbase - self.L) > 0.01:
            relative_error = self.L / current_wheelbase - 1.0
            self.xb += (self.xb - self.xf) * relative_error
            self.yb += (self.yb - self.yf) * relative_error

        # Update heading, psi.
        # --------------------
        delta_y = self.yf - self.yb
        if (self.xf == self.xb) and delta_y < 0.0:
            self.psi = np.pi
        else:
            if delta_y > 0.0:
                self.psi = arctan((self.xb - self.xf) / delta_y)
            else:
                self.psi = sign(self.xb - self.xf) * 0.5 * np.pi - arctan(delta_y / (self.xb - self.xf))

        # # Update angle to goal, psig (Lagoudakis, 2002, calls this "psi")
        # # --------------------
        # self.yg = self.y_goal
        # self.xg = self.x_goal
        # delta_yg = self.yg - self.yb
        # if (self.xg == self.xb) and delta_yg < 0.0:
        #     psig = psi - np.pi
        # else:
        #     if delta_y > 0.0:
        #         psig = psi - (arctan((xb - xg) / delta_yg))
        #     else:
        #         psig = psi - (sign(xb - xg) * 0.5 * np.pi - arctan(delta_yg / (xb - xg)))

        costs = 0.01*self.omega**2 + 0.1 * self.omegad ** 2 + self.omegadd ** 2
        # costs = self.omega ** 2 + 0.1 * self.omegad ** 2 + 0.01 * self.omegadd ** 2

        # costs = -1
        done = (self.omega > self.omega_threshold) or (self.omega < -self.omega_threshold)
        done = bool(done)

        return self._get_obs(), -costs, done, {}

    def _reset(self):
        self.theta = np.random.normal(0, 1) * np.pi / 180
        self.thetad = 0
        self.omega = np.random.normal(0, 1) * np.pi / 180
        self.omegad = 0
        self.omegadd = 0

        self.xb = 0
        self.yb = 0
        self.xf = self.xb + (np.random.rand(1)[0] * self.L - 0.5 * self.L)
        self.yf = np.sqrt(self.L ** 2 - (self.xf - self.xb) ** 2) + self.yb

        self.psi = np.arctan((self.xb - self.xf) / (self.yf - self.yb))
        # self.psig = self.psi - np.arctan((self.xb - self.x_goal) / (self.y_goal - self.yb))

        self.xfhist = []
        self.yfhist = []
        self.xbhist = []
        self.ybhist = []

        return self._get_obs()

    def get_xfhist(self):
        return self.xfhist

    def get_yfhist(self):
        return self.yfhist

    def get_xbhist(self):
        return self.xbhist

    def get_ybhist(self):
        return self.ybhist

    def _get_obs(self):
        return np.array([self.theta, self.thetad, self.omega, self.omegad, self.omegadd])

    def _render(self, mode='human', close=False):
        if close:
            self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(400, 400)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')