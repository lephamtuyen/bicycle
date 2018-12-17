import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign
from matplotlib import pyplot as plt

class BicycleGotoEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        # Environment parameters.
        self.time_step = 0.025

        # Goal position and radius
        # Lagouakis (2002) uses angle to goal, not heading, as a state
        self.r_goal = 1.
        # self.x_goal = 250.
        # self.y_goal = 0.
        # self.max_distance = 260.

        self.x_goal = 75.
        self.y_goal = 50.
        self.max_distance = 70.

        # Acceleration on Earth's surface due to gravity (m/s^2):
        self.g = 9.82

        # See the paper for a description of these quantities:
        # Distances (in meters):
        self.c = 0.66
        self.dCM = 0.30
        self.h = 0.94
        self.L = 1.11
        self.r = 0.34
        self.R1 = 0.136
        self.R2 = 0.068
        self.R3 = 0.034
        # Masses (in kilograms):
        self.Mc = 15.0
        self.Md = 1.7
        self.Mp = 60.0
        # Velocity of a bicycle (in meters per second), equal to 10 km/h:
        # self.v = 3.0 * 1000.0 / 3600.0

        # Derived constants.
        self.M = self.Mc + self.Mp  # See Randlov's code.
        self.Idc = self.Md * self.r ** 2
        self.Idv = 1.5 * self.Md * self.r ** 2
        self.Idl = 0.5 * self.Md * self.r ** 2
        self.Itot = 13.0 / 3.0 * self.Mc * self.h ** 2 + self.Mp * (self.h + self.dCM) ** 2

        # Angle at which to fail the episode
        self.omega_threshold = np.pi / 6
        self.theta_threshold = np.pi/2
        self.max_torque = 2.0
        self.max_displace = 0.2
        self.max_pedal = 100
        self.min_pedal = -100

        high = np.array([
            self.theta_threshold,
            np.finfo(np.float32).max,
            self.omega_threshold,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        ac_high = np.array([
            self.max_torque,
            self.max_displace,
            self.max_pedal])
        ac_low = np.array([
            -self.max_torque,
            -self.max_displace,
            self.min_pedal])
        self.action_space = spaces.Box(low=ac_low, high=ac_high)
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.reset()
        self.viewer = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        T = np.clip(u, -self.max_torque, self.max_torque)[0]
        d = np.clip(u, -self.max_displace, self.max_displace)[1]
        F = np.clip(u, self.min_pedal, self.max_pedal)[2]

        # store a last states
        self.last_xf = self.xf
        self.last_yf = self.yf
        self.last_omega = self.omega
        self.last_psig = self.psig

        # For recordkeeping.
        # ------------------
        self.xfhist.append(self.xf)
        self.yfhist.append(self.yf)
        self.xbhist.append(self.xb)
        self.ybhist.append(self.yb)
        self.vhist.append(self.v)
        self.shist.append(self._get_obs())

        if (self.omega > self.omega_threshold) or (self.omega < -self.omega_threshold):
            reward = -self.omega ** 2 - .1 * self.omegad ** 2 - 0.01 * self.omegadd ** 2 - 2.0 * self.psig ** 2  ###############2

            reach_goal = False
            done = False
            dist_to_goal = self.calc_dist_to_goal()
            if dist_to_goal == 0:
                print ('reached goal')
                reach_goal = True
                done = True

            done = bool(done)

            return self._get_obs(), reward, done, {"reach_goal": reach_goal}
        self.a = (F * self.R1* self.R3) / (self.R2 * self.r * (self.Mc+self.Md+self.Mp))
        self.v += self.a * self.time_step
        self.sigmad = self.v / self.r

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

        # Update angle to goal, psig (Lagoudakis, 2002, calls this "psi")
        self.yg = self.y_goal
        self.xg = self.x_goal
        delta_yg = self.yg - self.yb
        if (self.xg == self.xb) and delta_yg < 0.0:
            self.psig = self.psi - np.pi
        else:
            if delta_yg > 0.0:
                self.psig = self.psi - (arctan((self.xb - self.xg) / delta_yg))
            else:
                self.psig = self.psi - (
                sign(self.xb - self.xg) * 0.5 * np.pi - arctan(delta_yg / (self.xb - self.xg)))

        if (self.psig > np.pi):
            self.psig = self.psig - 2*np.pi
        elif(self.psig < -np.pi):
            self.psig = self.psig + 2*np.pi

        ###################################################
        # if np.abs(self.omega) > self.omega_threshold:
        #     reward = -1.0
        # else:
        #     distance = self.calc_dist_to_goal()
        #     # heading = self.calc_angle_to_goal()
        #     if (distance > self.max_distance):
        #         print ("MAX DISTANCE REACHED")
        #         reward = -1.0
        #     if (distance < 1e-3):
        #         print ("DEBUG: GOAL REACHED")
        #         reward = 0.01
        #     else:
        #         # reward from Randlov's 1998 paper
        #         reward = (4 - self.psig ** 2) * .00004
        ####################################################

        ####################################################
        # Lagoudakis reward function
        # delta_tilt = self.omega ** 2 - self.last_omega ** 2
        # delta_dist = self.calc_dist_to_goal() - self.calc_last_dist_to_goal()
        # reward = - delta_dist * 0.01
        # reward = -delta_tilt - delta_dist * 0.01
        # reward = -delta_tilt - delta_dist*0.001
        ####################################################

        ####################################################
        # delta_dist = self.calc_dist_to_goal() - self.calc_last_dist_to_goal()
        # reward = -0.01*self.omega ** 2 - .1 * self.omegad ** 2 - self.omegadd ** 2 - 10*delta_dist                 ###############2
        # reward = -self.omega ** 2 - .1 * self.omegad ** 2 - .01 * self.omegadd ** 2 - 0.01*delta_dist            #############1
        # reward = -self.omega ** 2 - .1 * self.omegad ** 2 - .01 * self.omegadd ** 2 - 0.01*self.calc_dist_to_goal()
        ####################################################
        reward = -self.omega ** 2 - .1 * self.omegad ** 2 - 0.01*self.omegadd ** 2 - 2.0*self.psig ** 2  ###############2

        reach_goal = False
        done = False
        #if np.abs(self.omega) > self.omega_threshold:
        #    done = True

        dist_to_goal = self.calc_dist_to_goal()
        if dist_to_goal == 0:
            print ('reached goal')
            reach_goal = True
            done = True

        #if dist_to_goal > self.max_distance:
        #    print ('greater than max distance')
        #    done = True

        done = bool(done)

        self.max_distance = np.min([self.max_distance, dist_to_goal*1.1])

        return self._get_obs(), reward, done, {"reach_goal":reach_goal}

    def _reset(self):
        self.theta = np.random.normal(0, 1) * np.pi / 180
        self.thetad = 0
        self.omega = np.random.normal(0, 1) * np.pi / 180
        self.omegad = 0
        self.omegadd = 0

        self.a = 0.0
        self.v = 0.0

        self.xb = 150 * np.random.rand(1)[0]
        self.yb = 100 * np.random.rand(1)[0]
        # self.xb = 0
        # self.yb = 0
        self.xf = self.xb + np.random.uniform(-1.0,1.0) * self.L
        self.yf = self.yb + np.sign(np.random.uniform(-1.0,1.0)) * \
                            np.sqrt(self.L ** 2 - (self.xf - self.xb) ** 2)

        # # Update heading, psi.
        # # --------------------
        # delta_y = self.yf - self.yb
        # delta_x = self.xf - self.xb
        # if (delta_y == 0.0):
        #     if delta_x < 0.0:
        #         self.psi = -np.pi / 2.0
        #     else:
        #         self.psi = np.pi / 2.0
        # elif (delta_y > 0):
        #     self.psi = arctan(delta_x / delta_y)
        # else:
        #     self.psi = sign(delta_x) * np.pi + arctan(delta_x / delta_y)

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

        # Update angle to goal, psig (Lagoudakis, 2002, calls this "psi")
        self.yg = self.y_goal
        self.xg = self.x_goal
        delta_yg = self.yg - self.yb
        if (self.xg == self.xb) and delta_yg < 0.0:
            self.psig = self.psi - np.pi
        else:
            if delta_yg > 0.0:
                self.psig = self.psi - (arctan((self.xb - self.xg) / delta_yg))
            else:
                self.psig = self.psi - (sign(self.xb - self.xg) * 0.5 * np.pi - arctan(delta_yg / (self.xb - self.xg)))

        if (self.psig > np.pi):
            self.psig = self.psig - 2 * np.pi
        elif (self.psig < -np.pi):
            self.psig = self.psig + 2 * np.pi
        # # Update angle to goal, psig (Lagoudakis, 2002, calls this "psi")
        # # --------------------
        # self.yg = self.y_goal
        # self.xg = self.x_goal
        # delta_yg = self.yg - self.yb
        # delta_xg = self.xg - self.xb
        # if (delta_yg == 0.0):
        #     if delta_xg < 0.0:
        #         self.psig = self.psi + np.pi / 2.0
        #     else:
        #         self.psig = self.psi - np.pi / 2.0
        # elif (delta_yg > 0):
        #     self.psig = self.psi - arctan(delta_xg / delta_yg)
        # else:
        #     self.psig = self.psi - (sign(delta_xg) * np.pi + arctan(delta_xg / delta_yg))
        #
        # if (self.psig > np.pi):
        #     self.psig = self.psig - 2*np.pi
        # elif(self.psig < -np.pi):
        #     self.psig = self.psig + 2*np.pi

        self.max_distance = self.calc_dist_to_goal() + 1

        self.xfhist = []
        self.yfhist = []
        self.xbhist = []
        self.ybhist = []
        self.vhist = []
        self.shist = []

        return self._get_obs()

    def get_xfhist(self):
        return self.xfhist

    def get_vhist(self):
        return self.vhist

    def get_shist(self):
        return self.shist

    def get_yfhist(self):
        return self.yfhist

    def get_xbhist(self):
        return self.xbhist

    def get_ybhist(self):
        return self.ybhist

    def _get_obs(self):
        return np.array([self.theta, self.thetad, self.omega, self.omegad, self.omegadd, self.psig])

    def _render(self, mode='human', close=False):
        def _render(self, mode='human', close=False):
            if close:
                self.viewer = None
                return

            if self.viewer is None:
                from gym.envs.classic_control import rendering
                self.viewer = rendering.Viewer(400, 400)

            return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def calc_dist_to_goal(self):
        # ported from Randlov's C code. See bike.c for the source
        # code.

        # unpack variables
        x_goal = self.x_goal
        y_goal = self.y_goal
        r_goal = self.r_goal
        xb = self.xb
        yb = self.yb

        sqrd_dist_to_goal = (x_goal - xb) ** 2 + (y_goal - yb) ** 2
        temp = np.max([0, sqrd_dist_to_goal - r_goal ** 2])

        # We probably don't need to actually compute a sqrt here if it
        # helps simulation speed.
        temp = np.sqrt(temp)

        return temp

    def calc_angle_to_goal(self):
        # ported from Randlov's C code. See bike.c for the source
        # code.

        # the following explanation of the returned angle is
        # verbatim from Randlov's C source:

        # These angles are neither in degrees nor radians, but
        # something strange invented in order to save CPU-time. The
        # measure is arranged same way as radians, but with a
        # slightly different negative factor
        #
        # Say the goal is to the east,
        # If the agent rides to the east then temp =  0
        #               " "         north    " "   = -1
        #               " "         west     " "   = -2 or 2
        #               " "         south    " "   =  1
        #
        # // end quote //


        # TODO: see the psi calculation in the environment, which is not
        # currently being used.


        # unpack variables
        x_goal = self.x_goal
        y_goal = self.y_goal
        xf = self.xf
        xb = self.xb
        yf = self.yf
        yb = self.yb

        # implement Randlov's angle computation
        # temp = (xf - xb) * (x_goal - xf) + (yf - yb) * (y_goal - yf)
        # scalar = temp / (1 * np.sqrt( (x_goal - xf)**2 + (y_goal - yf)**2))
        # tvaer = (-yf + yb) * (x_goal - xf) + (xf - xb) * (y_goal-yf)

        # if tvaer <= 0 :
        #    temp = scalar - 1
        # else:
        #    temp = np.abs(scalar - 1)

        # return temp

        # try just returning the angle in radians, instead of
        # randlov's funky units
        f2g = [(xf - x_goal), (yf - y_goal)]
        b2f = [(xf - xb), (yf - yb)]
        temp = np.dot(f2g, b2f) / (np.linalg.norm(f2g) * np.linalg.norm(b2f))
        # print temp
        temp = np.arccos(temp)

        return temp

    def calc_last_dist_to_goal(self):
        x_goal = self.x_goal
        y_goal = self.y_goal
        r_goal = self.r_goal

        last_xf = self.last_xf
        last_yf = self.last_yf

        sqrd_dist_to_goal = (x_goal - last_xf) ** 2 + (y_goal - last_yf) ** 2
        temp = np.max([0, sqrd_dist_to_goal - r_goal ** 2])

        return np.sqrt(temp)