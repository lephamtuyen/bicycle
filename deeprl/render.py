import cv2
import numpy as np
import gym
import time
import pickle
import numpy as np
np.random.seed(123)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
sns.set(color_codes=True)
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True


# ############################################################################################
# rewards1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/term1.npy")
# rewards2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/term2.npy")
# rewards3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/term3.npy")
# rewards4 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/term4.npy")
#
# plot_rewards1 = np.zeros([3, 80])
# plot_rewards2 = np.zeros([3, 80])
# plot_rewards3 = np.zeros([3, 80])
# plot_rewards4 = np.zeros([3, 80])
# for i in range(3):
#     for j in range(80):
#         plot_rewards1[i, j] = np.mean(rewards1[j * 10:(j + 1) * 10])
#         plot_rewards2[i, j] = np.mean(rewards2[j * 10:(j + 1) * 10])
#         plot_rewards3[i, j] = np.mean(rewards3[j * 10:(j + 1) * 10])
#         plot_rewards4[i, j] = np.mean(rewards4[j * 10:(j + 1) * 10])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
# sns_plot4 = sns.tsplot(data=plot_rewards4,color='y')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3,sns_plot4],
#            # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            labels=[r'$-\omega^2$',r'$-0.1{\dot(\omega)}^2$',r'$-0.01{\ddot(\omega)}^2$',r'$-2\psi_g^2$'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*10', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# rewards1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/1-fixed_goal_1_action/total_reward.npy")
# rewards2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/1-fixed_goal_2_action_2/total_reward.npy")
# rewards3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/1-fixed_goal_2_action_1/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 14])
# plot_rewards2 = np.zeros([3, 14])
# plot_rewards3 = np.zeros([3, 14])
# for i in range(3):
#     for j in range(14):
#         plot_rewards1[i, j] = np.mean(rewards1[j * 100:(j + 1) * 100])
#         plot_rewards2[i, j] = np.mean(rewards2[j * 100:(j + 1) * 100])
#         plot_rewards3[i, j] = np.mean(rewards3[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            labels=["d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')
#

# ############################################################################################
# import matplotlib.cm as cm
# x_back_trajectory = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/1-fixed_goal_2_action_1/back_trajectory.npy")
#
# plt.style.use('bmh')
# fig = plt.figure(1)
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# circle1 = plt.Circle((250, 0), 4, color='r')
# plt.gcf().gca().add_artist(circle1)
# plt.ylim([20, 120])
# plt.xlim([25, 175])
# trajectories = np.asarray(x_back_trajectory[0])
# colors = cm.rainbow(np.linspace(0, 1, 800))
# for episode in range(trajectories.shape[0]):
#     if episode % 5 == 0 and episode < 800:
#         current_plot = plt.plot(
#             trajectories[800 - episode - 1, 0],
#             trajectories[800 -episode - 1, 1], linewidth=1.0)
#
#         current_plot[0].set_color(colors[episode])
#
# # for episode in range(trajectories):
# #     plt.plot(trajectories[episode,:], trajectories[episode,:], linewidth=0.5, label='trajectory')
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# import matplotlib.cm as cm
# x_back_trajectory = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/2-fixed_goal_random_state_2_action/back_trajectory.npy")
#
# plt.style.use('bmh')
# fig = plt.figure(1)
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# circle1 = plt.Circle((250, 0), 4, color='r')
# plt.gcf().gca().add_artist(circle1)
# plt.ylim([0, 150])
# plt.xlim([0, 200])
# trajectories = np.asarray(x_back_trajectory[0])
# # colors = cm.rainbow(np.linspace(0, 1, 800))
# for episode in range(trajectories.shape[0]):
#     if episode >= 450 and episode < 600:
#         current_plot = plt.plot(
#             trajectories[episode, 0],
#             trajectories[episode, 1], linewidth=1.0)
#
#         current_plot[0].set_color('blue')
#
# # for episode in range(trajectories):
# #     plt.plot(trajectories[episode,:], trajectories[episode,:], linewidth=0.5, label='trajectory')
# fig.savefig("figure.pdf", bbox_inches='tight')

# # ############################################################################################
# reward = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/2-fixed_goal_random_state_2_action/total_reward.npy")
# step = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/2-fixed_goal_random_state_2_action/total_step.npy")
#
# from pandas import DataFrame
#
# plot_reward = np.zeros(120)
# plot_step = np.zeros(120)
# for j in range(120):
#     plot_reward[j] = np.mean(reward[j * 10:(j + 1) * 10])
#     plot_step[j] = np.mean(step[j * 10:(j + 1) * 10])
#
# data = np.column_stack((plot_reward, plot_step))
# df = DataFrame(data, columns=['A', 'B'])
# fig, ax = plt.subplots()
#
# df.A.plot(ax=ax, style='b-')
# # same ax as above since it's automatically added on the right
# df.B.plot(ax=ax, style='r-', secondary_y=True)
#
# plt.xlabel('Episodes*10', fontsize=16)
# # plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# # plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]], ['Reward','Steps'])
# fig.savefig("figure.pdf", bbox_inches='tight')


# ############################################################################################
# reward = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/3-random_goal_random_state_2_action/total_reward.npy")
# step = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/3-random_goal_random_state_2_action/total_step.npy")
#
# from pandas import DataFrame
#
# plot_reward = np.zeros(50)
# plot_step = np.zeros(50)
# for j in range(50):
#     plot_reward[j] = np.mean(reward[j * 100:(j + 1) * 100])
#     plot_step[j] = np.mean(step[j * 100:(j + 1) * 100])
#
# data = np.column_stack((plot_reward, plot_step))
# df = DataFrame(data, columns=['A', 'B'])
# fig, ax = plt.subplots()
#
# df.A.plot(ax=ax, style='b-')
# # same ax as above since it's automatically added on the right
# df.B.plot(ax=ax, style='r-', secondary_y=True)
#
# plt.xlabel('Episodes*100', fontsize=16)
# # plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# # plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# ax.legend([ax.get_lines()[0], ax.right_ax.get_lines()[0]], ['Reward','Steps'])
# fig.savefig("figure.pdf", bbox_inches='tight')


# ############################################################################################
# import matplotlib.cm as cm
# x_back_trajectory = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/3-random_goal_random_state_2_action/back_trajectory.npy")
#
# plt.style.use('bmh')
# fig = plt.figure(1)
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# circle1 = plt.Circle((250, 0), 4, color='r')
# plt.gcf().gca().add_artist(circle1)
# plt.ylim([0, 150])
# plt.xlim([0, 200])
# trajectories = np.asarray(x_back_trajectory[0])
# for episode in range(trajectories.shape[0]):
#     if episode >= 450 and episode < 600:
#         current_plot = plt.plot(
#             trajectories[episode, 0],
#             trajectories[episode, 1], linewidth=1.0)
#
#         current_plot[0].set_color('blue')
# fig.savefig("figure.pdf", bbox_inches='tight')

############################################################################################
import matplotlib.cm as cm
velocity = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/velocity.npy")

plt.style.use('bmh')
fig = plt.figure(1)
velocities = np.asarray(velocity[0])
for episode in range(velocity.shape[0]):
    if episode >= 13300:
        current_plot = plt.plot(velocity[episode], linewidth=1.0)

fig.savefig("figure.pdf", bbox_inches='tight')