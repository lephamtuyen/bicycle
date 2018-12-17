import gym
from agents.ddpg import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *
from matplotlib import pyplot as plt
from BicycleRender import BicycleRender

# Define environemnt name
ENV_NAME = "Bicycle-v0"

# Initialize the environment
env = ContinuousWrapper(gym.make(ENV_NAME))

# Get the action dimension and state dimension
action_dim = env.action_space.shape[0] # action_dim = 1
state_dim = env.observation_space.shape # state_dim = 5

# Initialize the network of DDPG algorithm
# online critic and target critic
critic = CriticNetwork(action_dim=action_dim, state_dim= state_dim)

# online actor and target actor
actor = ActorNetwork(action_dim=action_dim, state_dim= state_dim)

# initialize replay memory
# 64: batch size
memory = Memory(500000, state_dim, 1, 64)

def training(agent):
    # train the agent
    agent.train()

def testing(agent):
    agent.evaluate()

def simulate(agent):
    agent.restore()
    app = BicycleRender(agent, env)
    app.run()

with tf.Session() as sess:
    # create the agent and pass some parameters to the agent
    agent = DDPG(sess, actor, critic, memory, env=env,
                 max_test_epoch=10, warm_up=5000,
                 max_step_per_game=6000, is_plot=True,
                 render=False, max_episode=3000, env_name=ENV_NAME,
                 noise_theta=0.15, noise_sigma=0.1)

    # training(agent)
    # testing(agent)
    simulate(agent)




