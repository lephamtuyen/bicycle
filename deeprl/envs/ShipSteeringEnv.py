import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from numpy import sin, cos, tan, sqrt, arcsin, arctan, sign, exp
from matplotlib import pyplot as plt
from gym.envs.classic_control import rendering

SPEED = 3.0 #m/s
T = 5.0 # second
DELTA = 0.2 #second

MAX_TURNING_RATE = 15.0 * np.pi / 180.0
MIN_TURNING_RATE = -15.0 * np.pi / 180.0
MAX_THETA = np.pi
MIN_THETA = -np.pi
MAX_THETADOT = 15.0 * np.pi / 180.0
MIN_THETADOT = -15.0 * np.pi / 180.0

class ShipSteeringEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):

        self.bound = np.array([1000, 1000])
        self.gate = np.array([[800, 900], [900, 900]])

        self.goal = (self.gate[1] + self.gate[0]) / 2.0

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            MAX_THETA,
            MAX_THETADOT])
        self.action_space = spaces.Box(low=-MAX_TURNING_RATE, high=MAX_TURNING_RATE, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)

        self._seed()
        self.viewer = None
        self.subgoal = None

        self.xhist = []
        self.yhist = []

    def random_subgoal(self):
        self.initial_state = self._get_obs()

        t = 2 * np.pi * np.random.rand(1)[0]
        r = 50.
        xc = self.x
        yc = self.y

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        theta = MIN_THETA + (MAX_THETA - MIN_THETA) * np.random.rand(1)[0]
        thetadot = MIN_THETADOT + (MAX_THETADOT - MIN_THETADOT) * np.random.rand(1)[0]

        self.subgoal = np.array([x, y, theta, thetadot])

        return self.subgoal

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.x = self.bound[0]*np.random.rand(1)[0]
        self.y = self.bound[1]*np.random.rand(1)[0]

        self.theta = MIN_THETA + (MAX_THETA - MIN_THETA)*np.random.rand(1)[0]
        self.thetadot = MIN_THETADOT + (MAX_THETADOT - MIN_THETADOT)*np.random.rand(1)[0]

        self.last_xhist = self.xhist
        self.last_yhist = self.yhist

        self.xhist = []
        self.yhist = []

        return self._get_obs()

    def _step(self, action):
        turning_rate = np.clip(action, MIN_TURNING_RATE, MAX_TURNING_RATE)[0]

        # For recordkeeping.
        # ------------------
        self.xhist.append(self.x)
        self.yhist.append(self.y)

        # last position
        self.last_x = self.x
        self.last_y = self.y

        self.x = self.x + DELTA * SPEED * sin(self.theta)
        self.y = self.y + DELTA * SPEED * cos(self.theta)
        self.theta = self.theta + DELTA * self.thetadot

        if (self.theta > MAX_THETA):
            self.theta = -2*np.pi + self.theta
        elif (self.theta < MIN_THETA):
            self.theta = 2*np.pi + self.theta

        self.thetadot = self.thetadot + DELTA * (turning_rate - self.thetadot) / T
        self.thetadot = np.clip(self.thetadot, MIN_THETADOT, MAX_THETADOT)

        # if (isPassed==True):
        #     reward = 0
        # else:
        #     reward = -1

        # reward = - np.sqrt((self.x-self.goal[0])**2 + (self.y-self.goal[1])**2)


        # if (self.x>self.bound[0] or self.x<0 or self.y>self.bound[1] or self.y<0):
        #     reward = -100

        # if (isPassed):
        #     reward = 100
        # else:
        #     angle_to_goal = self.calc_angle_to_goal(np.array([self.x,self.y]), self.goal)
        #     reward = exp(- ((self.theta-angle_to_goal)**2)/900.0) - 1


        reward = self.getReward(self.goal)

        if (self.subgoal is not None):
            intrinsic_reward = self.getReward(self.subgoal[:2])

        isOut, isReach = self.checkSubgoal()

        isPassed = self.checkPassThroughTheGate(np.array([self.last_x, self.last_y]), np.array([self.x, self.y]))

        done = self.x>self.bound[0] or self.x<0 or self.y>self.bound[1] or self.y<0 or isPassed==True
        done = bool(done)

        if (done == True):
            self.last_xhist = self.xhist
            self.last_yhist = self.yhist

        info = {}
        if (isPassed == True):
            print ('gate passed')
            info["reach_goal"] = True
        else:
            info["reach_goal"] = False

        if (isReach is not None):
            if (isReach == True):
                print ('reach subgoal')
                info["reach_subgoal"] = True
            else:
                info["reach_subgoal"] = False

        if (isOut is not None):
            if (isOut == True):
                print ('out of subgoal')
                info["out_of_subgoal"] = True
            else:
                info["out_of_subgoal"] = False

        if (self.subgoal is not None):
            info["intrinsic_reward"] = intrinsic_reward

        return self._get_obs(), reward, done, info

    def getReward(self, target):
        ###############################################################################
        dist_to_goal = np.sqrt((self.x - target[0]) ** 2 + (self.y - target[1]) ** 2)
        last_dist_to_goal = np.sqrt((self.last_x - target[0]) ** 2 + (self.last_y - target[1]) ** 2)
        delta_dist = dist_to_goal - last_dist_to_goal

        angle_to_goal = self.calc_angle_to_goal(np.array([self.x, self.y]), target)
        reward = -delta_dist / 0.3 + exp(- (((self.theta - angle_to_goal) / (2 * np.pi)) ** 2))
        # reward = -10*delta_dist + exp(- (((self.theta - angle_to_goal) / (2 * np.pi)) ** 2))
        ###############################################################################

        return reward

    def _render(self, mode='human', close=False):
        if close:
            self.viewer = None
            return

        screen_width = 1000
        screen_height = 1000

        world_width = self.bound[0]
        scale = screen_width / world_width
        shipwidth = 20
        shipheight = 60
        goal_radius = 10
        subgoal_radius = 5

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            point1 = (-shipwidth/2, 0)
            point2 = (-shipwidth / 2., 2.0*shipheight/3.0)
            point3 = (0., shipheight)
            point4 = (shipwidth / 2., 2.0*shipheight/3.0)
            point5 = (shipwidth / 2, 0)
            ship = rendering.FilledPolygon([point1, point2, point3, point4, point5])
            self.shiptrans = rendering.Transform()
            ship.add_attr(self.shiptrans)
            ship.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(ship)

            goal = rendering.make_circle(goal_radius * scale)
            goal.add_attr(rendering.Transform(translation=(self.goal[0] * scale, self.goal[1] * scale)))
            goal.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(goal)

        if (self.subgoal is not None):
            subgoal = rendering.make_circle(subgoal_radius * scale)
            subgoal.add_attr(rendering.Transform(translation=(self.subgoal[0] * scale, self.subgoal[1] * scale)))
            subgoal.set_color(1.0, 1.0, 0.0)
            self.viewer.add_geom(subgoal)

        # pos = self.state[0]
        self.shiptrans.set_translation(self.x * scale, self.y * scale)
        self.shiptrans.set_rotation(-self.theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _get_obs(self):
        return np.array([self.x, self.y, self.theta, self.thetadot])

    def checkSubgoal(self):
        if (self.subgoal is not None):
            dist_to_goal = np.sqrt((self.x - self.subgoal[0]) ** 2 + (self.y - self.subgoal[1]) ** 2)
            dist_to_initial = np.sqrt((self.x - self.initial_state[0]) ** 2 + (self.y - self.initial_state[1]) ** 2)

            return dist_to_initial > 50.0, dist_to_goal < 5.0
        else:
            return None, None

    def checkPassThroughTheGate(self, start, end):
        r = self.gate[1] - self.gate[0]
        s = end - start

        den = self.cross2D(r, s);

        if (den == 0):
            return False;

        t = self.cross2D((start - self.gate[0]), s) / den;

        u = self.cross2D((start - self.gate[0]), r) / den;

        return u >= 0 and u <= 1 and t >= 0 and t <= 1;

    def cross2D(self, v, w):
        return v[0] * w[1] - v[1] * w[0]

    def get_xhist(self):
        return self.last_xhist

    def get_yhist(self):
        return self.last_yhist

    def calc_angle_to_goal(self, current_position, goal):

        vt = goal - current_position

        if (vt[1] == 0):
            if (vt[0] < 0):
                return -np.pi/2.0
            else:
                return np.pi/2.0
        else:
            angle = arctan(vt[0]/vt[1]);
            if (vt[1] > 0):
                return angle
            else:
                return sign(vt[0])*np.pi + angle

# xxx = ShipSteeringEnv()
# xxx.reset()
# for i in range(1,1000,1):
#     state, reward, done, _ = xxx.step(np.array([0.0]))
#
#     if (done == True):
#         break
#
# plt.style.use('bmh')
# plt.figure(1)
# plt.ylim([0, 1000])
# plt.xlim([0, 1000])
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# plt.plot(xxx.xhist, xxx.yhist, linewidth=0.5, label='trajectory')
# plt.pause(1)
# input("Press Enter to end...")
