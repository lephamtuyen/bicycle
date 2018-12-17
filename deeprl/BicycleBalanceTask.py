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

with tf.Session() as sess:
    # create the agent and pass some parameters to the agent
    agent = DDPG(sess, actor, critic, memory, env=env,
                 max_test_epoch=10, warm_up=5000,
                 max_step_per_game=6000, is_plot=True,
                 render=False, max_episode=3000, env_name=ENV_NAME,
                 noise_theta=0.15, noise_sigma=0.1)

    # train the agent
    # agent.train()


    agent.evaluate()


    # agent.restore()
    # # Draw plot
    # plt.style.use('bmh')
    # plt.ion()
    # plt.figure(1)
    # cummulated_reward = []
    # for epoch in range(5):
    #     state = env.reset()
    #     step = 0
    #     done = False
    #     epi_reward = 0
    #     while not done:
    #         state = state[np.newaxis]
    #         action = agent.action(state)
    #         state, reward, done, _ = env.step(action.flatten())
    #         step += 1
    #         epi_reward += reward
    #
    #         if (step > 10000): done = True
    #
    #     cummulated_reward.append(epi_reward)
    #     back_lines = plt.plot(env.env.get_xbhist(), env.env.get_ybhist(), linewidth=0.5)
    #     plt.axis('equal')
    #     plt.pause(0.001)
    #
    #     plt.xlabel('Distances (m)')
    #     plt.ylabel('Distances (m)')
    #     agent.save_plot_figure(plt, 'evaluate_trajectory.pdf')
    # input("Press Enter to end...")

    # agent.restore()
    # app = BicycleRender(agent, env)
    # app.run()

    # total_reward = agent.restore_plot_data("total_reward.npy")
    # total_step = agent.restore_plot_data("total_step.npy")
    # x_back_trajectory = agent.restore_plot_data("x_back_trajectory.npy")
    # x_front_trajectory = agent.restore_plot_data("x_front_trajectory.npy")
    # y_back_trajectory = agent.restore_plot_data("y_back_trajectory.npy")
    # y_front_trajectory = agent.restore_plot_data("y_front_trajectory.npy")

    # plt.style.use('bmh')
    # plt.figure(1)
    # plt.xlabel('Distances (m)')
    # plt.ylabel('Distances (m)')
    # for episode in range(x_back_trajectory.shape[0]):
    #     plt.plot(x_back_trajectory[episode,], y_back_trajectory[episode,], linewidth=0.5, label='trajectory')
    # # plt.ylim([args.ymin, args.ymax])
    # # plt.legend()
    # agent.save_plot_figure(plt,'train_trajectory.pdf')

    # plt.style.use('bmh')
    # plt.figure(2)
    # plt.xlabel('Episode')
    # plt.ylabel('Reward')
    #
    # xxx = total_reward.shape[0]
    # epis = np.arange(0, total_reward.shape[0], 1)
    # plt.plot(epis, total_reward, label='reward',linewidth=1.0)
    # # plt.ylim([args.ymin, args.ymax])
    # # plt.legend()
    # agent.save_plot_figure(plt,'reward.pdf')
    #
    # plt.style.use('bmh')
    # plt.figure(3)
    # plt.xlabel('Episode')
    # plt.ylabel('Steps')
    # xxx = total_step.shape[0]
    # epis = np.arange(0, total_step.shape[0], 1)
    # plt.plot(epis, total_step, label='steps',linewidth=1.0)
    # # plt.ylim([args.ymin, args.ymax])
    # # plt.legend()
    # agent.save_plot_figure(plt, 'steps.pdf')