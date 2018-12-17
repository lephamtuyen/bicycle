import gym
from agents.ddpg import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *
import sys
import time
from math import pi, sin, cos
import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.showbase import DirectObject
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode

class BicycleRender(ShowBase):

    rad2deg = 180. / 3.14
    def __init__(self,agent=None,env=None,show_learning=True):
        ShowBase.__init__(self)
        ShowBase.movie(self, namePrefix='images/camera1/jmage',
                       duration=60,
                       fps=60,
                       format='jpg',
                       sd=4,
                       source=None)

        self.agent = agent
        self.env=env
        self.bicycle = self.env.env
        current_state = self.env.reset()
        current_state = current_state[np.newaxis]
        self.action = self.agent.action(current_state)
        self.action = self.action.flatten()

        self.show_learning = show_learning
        if self.show_learning:
            self.theta_counter = 0
            self.theta_index = 0
        self.elapsed_time = 0

        self.omegaText = self.genLabelText("", 1)
        self.thetaText = self.genLabelText("", 2)
        self.timeText = self.genLabelText("", 3)

        self.wheel_roll = 0
        self.torque = 0
        self.butt_displacement = 0

        # Load the environment model.
        self.environ = self.loader.loadModel("maps/Ground2.egg")
        ## Reparent the model to render.
        self.environ.reparentTo(self.render)

        # Disable the use of the mouse to control the camera.
        self.disableMouse()

        # "out-of-body experience"; toggles camera control.
        self.accept('o', self.oobe)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.followBikeTask, "FollowBikeTask")
        self.taskMgr.add(self.simulateBicycleTask, "SimulateBicycleTask")

        self.rear_wheel = self.loader.loadModel("maps/wheel3.egg")
        self.rear_wheel.reparentTo(self.render)
        self.rear_wheel.setPos(0, 0, self.bicycle.r)

        self.frame = self.loader.loadModel("maps/frame.egg")
        self.frame.reparentTo(self.rear_wheel)
        self.frame.setColor(1, 0, 0)

        self.butt = self.loader.loadModel("maps/frame.egg")
        self.butt.reparentTo(self.frame)
        self.butt.setColor(1, 0, 0)
        self.butt.setScale(1, 0.1, 1)
        self.butt.setZ(1.5 * self.bicycle.r)
        self.butt.setY(0.3 * self.bicycle.L)

        # place goal
        self.goalPost = self.loader.loadModel("maps/fork.egg")
        self.goalPost.reparentTo(self.render)

        self.fork = self.loader.loadModel("maps/fork.egg")
        self.fork.reparentTo(self.frame)
        self.fork.setColor(0, 0, 1)
        self.fork.setPos(0, self.bicycle.L, self.bicycle.r)

        self.front_wheel = self.loader.loadModel("maps/wheel3.egg")
        self.front_wheel.reparentTo(self.fork)
        self.front_wheel.setColor(1, 1, 1)
        self.front_wheel.setPos(0, 0, -self.bicycle.r)

        self.handlebar = self.loader.loadModel("maps/fork.egg")
        self.handlebar.reparentTo(self.fork)
        self.handlebar.setColor(0, 0, 1)
        self.handlebar.setPos(0, 0, self.bicycle.r)
        self.handlebar.setHpr(0, 0, 90)

        self.torqueLeftIndicator = self.loader.loadModel("maps/fork.egg")
        self.torqueLeftIndicator.reparentTo(self.fork)
        self.torqueLeftIndicator.setColor(0, 0, 1)
        self.torqueLeftIndicator.setPos(-self.bicycle.r, 0, self.bicycle.r)
        self.torqueLeftIndicator.hide()

        self.torqueRightIndicator = self.loader.loadModel("maps/fork.egg")
        self.torqueRightIndicator.reparentTo(self.fork)
        self.torqueRightIndicator.setColor(0, 0, 1)
        self.torqueRightIndicator.setPos(self.bicycle.r, 0, self.bicycle.r)
        self.torqueRightIndicator.hide()

        self.camera.setPos(5, -5, 10)

    # Define a procedure to move the camera.
    def followBikeTask(self, task):
        #camera 1
        look = self.rear_wheel.getPos()
        self.camera.lookAt(look[0], look[1], look[2] + 1.0)
        self.camera.setPos(look[0] - 1.0, look[1] - 6.0, look[2] + 2.0)

        return Task.cont

    #Macro-like function used to reduce the amount to code needed to create the
    #on screen instructions
    def genLabelText(self, text, i, scale=None):
        if scale == None:
            scale = 0.05
        textObject = OnscreenText(text = text, pos = (-1.3, .95-.05*i), fg=(1,1,0,1),
                      align = TextNode.ALeft, scale = scale)
        return textObject

    def simulateBicycleTask(self, task):
        self.elapsed_time += self.bicycle.time_step

        elapsedstr = "Elapsed time = %3.3f" % (self.elapsed_time)
        tiltstr = "Omega = %3.3f" % (self.bicycle.omega*180/np.pi)
        thetastr = "Theta = %3.3f" % (self.bicycle.theta*180/np.pi)
        self.omegaText.setText(tiltstr)
        self.thetaText.setText(thetastr)
        self.timeText.setText(elapsedstr)

        state, reward, done, _ = self.env.step(self.action)
        state = state[np.newaxis]

        if not done:
            self.wheel_roll += self.bicycle.time_step * self.bicycle.sigmad
            self.rear_wheel.setPos(self.bicycle.xb, self.bicycle.yb, self.bicycle.r)
            self.rear_wheel.setP(-self.rad2deg * self.wheel_roll)
            self.rear_wheel.setR(self.rad2deg * self.bicycle.omega)


            self.frame.setP(self.rad2deg * self.wheel_roll)
            self.butt.setX(0)
            self.fork.setH(self.rad2deg * self.bicycle.theta)

            self.front_wheel.setP(-self.rad2deg * self.wheel_roll)

            self.action = self.agent.action(state)
            self.action = self.action.flatten()
        else:
            time.sleep(5)
            state=self.env.reset()
            state = state[np.newaxis]
            self.action = self.agent.action(state)
            self.action = self.action.flatten()
            self.elapsed_time = 0

        return Task.cont



