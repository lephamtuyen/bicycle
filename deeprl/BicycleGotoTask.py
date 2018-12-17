import gym
from agents.ddpg import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *
#from BicycleRender import BicycleRender
from matplotlib import pyplot as plt

ENV_NAME = "Bicycle-v1"

env = ContinuousWrapper(gym.make(ENV_NAME))
action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape
critic = CriticNetwork(action_dim=action_dim, state_dim= state_dim)
actor = ActorNetwork(action_dim=action_dim, state_dim= state_dim)
memory = Memory(500000, state_dim, action_dim, 64)

with tf.Session() as sess:
    agent = DDPG(sess, actor, critic, memory, env=env, max_test_epoch=20,warm_up=1000,max_step_per_game = 400,is_plot=True,
                 render=False, max_episode=5000, env_name=ENV_NAME,noise_theta=0.15,noise_sigma=0.1)
    agent.train()
    # agent.evaluate()

    # agent.restore()
    # # Draw plot
    # plt.style.use('bmh')
    # plt.ion()
    # plt.figure(1)
    # cummulated_reward = []
    # for epoch in range(100):
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
    #         if (step > 4000): done = True
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
    #
    # plt.style.use('bmh')
    # plt.figure(1)
    # plt.xlabel('Distances (m)')
    # plt.ylabel('Distances (m)')
    # circle1 = plt.Circle((250, 0), 10, color='r')
    # plt.gcf().gca().add_artist(circle1)
    # plt.ylim([-150, 250])
    # plt.xlim([-10, 260])
    # for episode in range(x_back_trajectory.shape[0]):
    #     if (len(x_front_trajectory[episode,]) > 5000):
    #         plt.plot(x_front_trajectory[episode,], y_back_trajectory[episode,], linewidth=0.5, label='trajectory')
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

    # plt.style.use('bmh')
    # plt.figure(1)
    # plt.xlabel('Distances (m)')
    # plt.ylabel('Distances (m)')
    # circle1 = plt.Circle((250, 0), 10, color='r')
    # plt.gcf().gca().add_artist(circle1)
    # plt.ylim([-150, 250])
    # plt.xlim([-10, 260])
    # for episode in range(x_back_trajectory.shape[0]):
    #     if (len(x_front_trajectory[episode,]) < 100):
    #         plt.plot(x_front_trajectory[episode,], y_back_trajectory[episode,], linewidth=0.5, label='trajectory')
    # plt.ylim([args.ymin, args.ymax])
    # plt.legend()
    # agent.save_plot_figure(plt, 'initial_position.pdf')