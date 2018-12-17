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
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


# ############################################################################################
# term11 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/term1.npy")
# term21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/term2.npy")
# term31 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/term3.npy")
# term41 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/term4.npy")
# term12 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/term1.npy")
# term22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/term2.npy")
# term32 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/term3.npy")
# term42 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/term4.npy")
# term13 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/term1.npy")
# term23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/term2.npy")
# term33 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/term3.npy")
# term43 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/term4.npy")
#
# plot_rewards1 = np.zeros([3, 50])
# plot_rewards2 = np.zeros([3, 50])
# plot_rewards3 = np.zeros([3, 50])
# plot_rewards4 = np.zeros([3, 50])
# for j in range(50):
#     plot_rewards1[0, j] = np.mean(term11[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(term12[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(term13[j * 100:(j + 1) * 100])
#
#     plot_rewards2[0, j] = np.mean(term21[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(term22[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(term23[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(term31[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(term32[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(term33[j * 100:(j + 1) * 100])
#
#     plot_rewards4[0, j] = np.mean(term41[j * 100:(j + 1) * 100])
#     plot_rewards4[1, j] = np.mean(term42[j * 100:(j + 1) * 100])
#     plot_rewards4[2, j] = np.mean(term43[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
# sns_plot4 = sns.tsplot(data=plot_rewards4,color='y')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3,sns_plot4],
#            bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            labels=[r'$-\omega^2$',r'$-0.1{(\dot{\omega})}^2$',r'$-0.01{(\ddot{\omega})}^2$',r'$-2\psi_g^2$'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ###########################################################################################
# rewards11 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_1/total_reward.npy")
# rewards12 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_2/total_reward.npy")
# rewards13 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_3/total_reward.npy")
# rewards21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_1/total_reward.npy")
# rewards22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_2/total_reward.npy")
# rewards23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_3/total_reward.npy")
# rewards31 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# rewards32 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# rewards33 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 50])
# plot_rewards2 = np.zeros([3, 50])
# plot_rewards3 = np.zeros([3, 50])
# for j in range(50):
#     plot_rewards1[0, j] = np.mean(rewards11[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(rewards12[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(rewards13[j * 100:(j + 1) * 100])
#
#     plot_rewards2[0, j] = np.mean(rewards21[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(rewards22[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(rewards23[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(rewards31[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(rewards32[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(rewards33[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0.6, 0.25), loc=2, borderaxespad=0.,
#            labels=["d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')


############################################################################################
# import matplotlib.cm as cm
# x_back_trajectory_1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_1/back_trajectory.npy")
# x_back_trajectory_2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_1/back_trajectory.npy")
# x_back_trajectory_3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/back_trajectory.npy")
#
# plt.style.use('bmh')
# fig = plt.figure(1)
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# # circle1 = plt.Circle((250, 0), 4, color='r')
# # plt.gcf().gca().add_artist(circle1)
# plt.ylim([40, 70])
# plt.xlim([40, 70])
# trajectories = np.asarray(x_back_trajectory_3[0])
# colors = cm.rainbow(np.linspace(0, 1, 5000))
# random_draw_sequence = np.random.permutation(range(trajectories.shape[0]))
# for episode in range(trajectories.shape[0]):
#     current_plot = plt.plot(
#             trajectories[random_draw_sequence[episode], 0],
#             trajectories[random_draw_sequence[episode], 1], linewidth=0.1)
#
#     current_plot[0].set_color(colors[random_draw_sequence[episode]])
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# ddpg1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ddpg2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ddpg3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# ppo1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ppo2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ppo3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# naf1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# naf2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# naf3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 49])
# plot_rewards2 = np.zeros([3, 49])
# plot_rewards3 = np.zeros([3, 49])
# for j in range(49):
#     plot_rewards1[0, j] = np.mean(ddpg1[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(ddpg2[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(ddpg3[j * 100:(j + 1) * 100])
#     #
#     plot_rewards2[0, j] = np.mean(ppo1[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(ppo2[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(ppo3[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(naf1[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(naf2[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(naf3[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.,
#            labels=[r'DDPG',r'PPO',r'NAF'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ###########################################################################################
# ddpg1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
# ddpg2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_2/total_reward.npy")
# ddpg3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_3/total_reward.npy")
# ppo1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
# ppo2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_dynamic_velocity_d_0.02_trial_2/total_reward.npy")
# ppo3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_dynamic_velocity_d_0.02_trial_3/total_reward.npy")
# naf1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
# naf2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
# naf3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 99])
# plot_rewards2 = np.zeros([3, 99])
# plot_rewards3 = np.zeros([3, 99])
# for j in range(99):
#     plot_rewards1[0, j] = np.mean(ddpg1[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(ddpg2[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(ddpg3[j * 100:(j + 1) * 100])
#     #
#     plot_rewards2[0, j] = np.mean(ppo1[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(ppo2[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(ppo3[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(naf1[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(naf2[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(naf3[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.,
#            labels=[r'DDPG',r'PPO',r'NAF'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# rewards11 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/total_reward.npy")
# rewards12 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_2/total_reward.npy")
# rewards13 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_3/total_reward.npy")
# rewards21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/total_reward.npy")
# rewards22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_2/total_reward.npy")
# rewards23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_3/total_reward.npy")
#
# x_front_11 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/front_trajectory.npy")
# x_front_12 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_2/front_trajectory.npy")
# x_front_13 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_3/front_trajectory.npy")
# x_front_21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/front_trajectory.npy")
# x_front_22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_2/front_trajectory.npy")
# x_front_23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_3/front_trajectory.npy")
# x_front_11 = np.asarray(x_front_11[0])
# x_front_12 = np.asarray(x_front_12[0])
# x_front_13 = np.asarray(x_front_13[0])
# x_front_21 = np.asarray(x_front_21[0])
# x_front_22 = np.asarray(x_front_22[0])
# x_front_23 = np.asarray(x_front_23[0])
#
# plot_rewards1 = np.zeros([3, 100])
# plot_rewards2 = np.zeros([3, 100])
# plot_rewards3 = np.zeros([3, 100])
# for j in range(100):
#     i11 = 0
#     i12 = 0
#     i13 = 0
#     i21 = 0
#     i22 = 0
#     i23 = 0
#     plot_rewards1[0, j] = 0
#     plot_rewards1[1, j] = 0
#     plot_rewards1[2, j] = 0
#     plot_rewards2[0, j] = 0
#     plot_rewards2[1, j] = 0
#     plot_rewards2[2, j] = 0
#     for k in range(j * 100,(j + 1) * 100):
#         x = x_front_11[k, 0][0]
#         y = x_front_11[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards1[0, j] += rewards11[k]
#             i11 += 1
#
#         x = x_front_12[k, 0][0]
#         y = x_front_12[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards1[1, j] += rewards12[k]
#             i12 += 1
#
#         x = x_front_13[k, 0][0]
#         y = x_front_13[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards1[2, j] += rewards13[k]
#             i13 += 1
#
#         x = x_front_21[k, 0][0]
#         y = x_front_21[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards2[0, j] += rewards21[k]
#             i21 += 1
#
#         x = x_front_22[k, 0][0]
#         y = x_front_22[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards2[1, j] += rewards22[k]
#             i22 += 1
#
#         x = x_front_23[k, 0][0]
#         y = x_front_23[k, 1][0]
#         if (x < 50 and y < 50):
#             plot_rewards2[2, j] += rewards23[k]
#             i23 += 1
#
#     plot_rewards1[0, j] /= i11
#     plot_rewards1[1, j] /= i12
#     plot_rewards1[2, j] /= i13
#     plot_rewards2[0, j] /= i21
#     plot_rewards2[1, j] /= i22
#     plot_rewards2[2, j] /= i23
#
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2],
#            # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            labels=["dynamic velocity","fixed velocity"],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

############################################################################################
# import matplotlib.cm as cm
# x_back_trajectory_1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/back_trajectory.npy")
# x_back_trajectory_2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/back_trajectory.npy")
# x_front_trajectory_1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/front_trajectory.npy")
# x_front_trajectory_2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/front_trajectory.npy")
#
# plt.style.use('bmh')
# fig = plt.figure(1)
# plt.xlabel('Distances (m)')
# plt.ylabel('Distances (m)')
# # circle1 = plt.Circle((250, 0), 4, color='r')
# # plt.gcf().gca().add_artist(circle1)
# plt.ylim([40, 70])
# plt.xlim([40, 70])
# trajectories = np.asarray(x_back_trajectory_1[0])
# front_trajectories = np.asarray(x_front_trajectory_1[0])
# colors = cm.rainbow(np.linspace(0, 1, 10000))
# random_draw_sequence = np.random.permutation(range(10000))
# for episode in range(10000):
#     x = front_trajectories[random_draw_sequence[episode], 0][0]
#     y = front_trajectories[random_draw_sequence[episode], 1][0]
#     if (x < 50 and y < 50):
#         current_plot = plt.plot(
#             trajectories[random_draw_sequence[episode], 0],
#             trajectories[random_draw_sequence[episode], 1], linewidth=0.1)
#
#         current_plot[0].set_color(colors[random_draw_sequence[episode]])
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# import matplotlib.cm as cm
# velocities = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/velocity.npy")
# step = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/total_step.npy")
# front = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/front_trajectory.npy")
# front = np.asarray(front[0])
# plt.style.use('bmh')
# fig = plt.figure(1)
#
# opposite_velocity = []
# same_velocity = []
#
# for episode in range(velocities.shape[0]):
#     x = front[episode, 0][0]
#     y = front[episode, 1][0]
#     if (step[episode] < 500 and episode > 7000):
#         if(x < 50 and y < 50):
#             opposite_velocity.append(velocities[episode])
#         if(x > 50 and y > 50):
#             same_velocity.append(velocities[episode])
#
# opposite_velocity_array = np.zeros([len(opposite_velocity), 500])
# same_velocity_array = np.zeros([len(same_velocity), 500])
# for episode in range(opposite_velocity_array.shape[0]):
#     velocity = opposite_velocity[episode]
#     for i in range(500):
#         if (i < len(velocity)):
#             opposite_velocity_array[episode, i] = velocity[i]
#         else:
#             opposite_velocity_array[episode, i] = velocity[len(velocity)-1]
# for episode in range(same_velocity_array.shape[0]):
#     velocity = same_velocity[episode]
#     for i in range(500):
#         if (i < len(velocity)):
#             same_velocity_array[episode, i] = velocity[i]
#         else:
#             same_velocity_array[episode, i] = velocity[len(velocity)-1]
#
# sns_plot1 = sns.tsplot(data=opposite_velocity_array,color='r')
# sns_plot2 = sns.tsplot(data=same_velocity_array,color='g')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2],
#            # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            labels=["opposite direction","same direction"],
#            prop={'size':16})
#
# plt.xlabel('Time steps', fontsize=16)
# plt.ylabel('Velocity', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ############################################################################################
# import matplotlib.cm as cm
# velocities = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_2/velocity.npy")
# plt.style.use('bmh')
# fig = plt.figure(1)
#
# velocity = []
#
# for episode in range(velocities.shape[0]):
#     if (episode > 8000):
#         velocity.append(velocities[episode])
#
# velocity_array = np.zeros([len(velocity), 1000])
# for episode in range(velocity_array.shape[0]):
#     v = velocity[episode]
#     for i in range(1000):
#         if (i < len(v)):
#             velocity_array[episode, i] = v[i]
#         else:
#             velocity_array[episode, i] = v[len(v)-1]
#
# sns_plot1 = sns.tsplot(data=velocity_array,color='r')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1],
#            # bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
#            # labels=["opposite direction"],
#            prop={'size':16})
#
# plt.xlabel('Time steps', fontsize=16)
# plt.ylabel('Velocity', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# # ############################################################################################
# reward1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_1/total_reward.npy")
# reward2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_2/total_reward.npy")
# reward3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_3/total_reward.npy")
# step1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_1/total_step.npy")
# step2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_2/total_step.npy")
# step3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_3/total_step.npy")
# goal1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_1/goal.npy")
# goal2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_2/goal.npy")
# goal3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_3/goal.npy")
# back_wheel_1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_1/back_trajectory.npy")
# back_wheel_2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_2/back_trajectory.npy")
# back_wheel_3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/final_trial_3/back_trajectory.npy")
# back_wheel_1 = np.asarray(back_wheel_1[0])
# back_wheel_2 = np.asarray(back_wheel_2[0])
# back_wheel_3 = np.asarray(back_wheel_3[0])
# distance = np.zeros([3,9000])
# for i in range(9000):
#     sss = back_wheel_1.shape[0]-1
#     x = back_wheel_1[i, 0][len(back_wheel_1[i, 0])-1]
#     y = back_wheel_1[i, 1][len(back_wheel_1[i, 0])-1]
#     g = goal1[i]
#     distance[0,i] = np.sqrt((x - g[0])**2 + (y - g[1])**2)
#
#     x = back_wheel_2[i, 0][len(back_wheel_2[i, 0])-1]
#     y = back_wheel_2[i, 1][len(back_wheel_2[i, 0])-1]
#     g = goal2[i]
#     distance[1,i] = np.sqrt((x - g[0]) ** 2 + (y - g[1]) ** 2)
#
#     x = back_wheel_3[i, 0][len(back_wheel_3[i, 0])-1]
#     y = back_wheel_3[i, 1][len(back_wheel_3[i, 0])-1]
#     g = goal3[i]
#     distance[2,i] = np.sqrt((x - g[0]) ** 2 + (y - g[1]) ** 2)
#
# plot_rewards1 = np.zeros([3, 90])
# for j in range(90):
#     plot_rewards1[0, j] = np.mean(distance[0,j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(distance[1,j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(distance[2,j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1],
#            bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.,
#            labels=[r'DDPG'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Meter', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')

# ###########################################################################################
# rewards11 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_1/total_reward.npy")
# rewards12 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_2/total_reward.npy")
# rewards13 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_3/total_reward.npy")
# rewards21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_1/total_reward.npy")
# rewards22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_2/total_reward.npy")
# rewards23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_3/total_reward.npy")
# rewards31 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# rewards32 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# rewards33 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 50])
# plot_rewards2 = np.zeros([3, 50])
# plot_rewards3 = np.zeros([3, 50])
# for j in range(50):
#     plot_rewards1[0, j] = np.mean(rewards11[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(rewards12[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(rewards13[j * 100:(j + 1) * 100])
#
#     plot_rewards2[0, j] = np.mean(rewards21[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(rewards22[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(rewards23[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(rewards31[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(rewards32[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(rewards33[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0.6, 0.25), loc=2, borderaxespad=0.,
#            labels=["d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')


# ###########################################################################################
# states = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/states/printed_states.npy")
#
# plot_rewards1 = np.zeros([3, 50])
# plot_rewards2 = np.zeros([3, 50])
# plot_rewards3 = np.zeros([3, 50])
# for j in range(50):
#     plot_rewards1[0, j] = np.mean(rewards11[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(rewards12[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(rewards13[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0.6, 0.25), loc=2, borderaxespad=0.,
#            labels=["d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')


# ############################################################################################
# ddpg1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ddpg2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ddpg3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# ppo1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ppo2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ppo3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# naf1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# naf2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# naf3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
# plot_rewards1 = np.zeros([3, 49])
# plot_rewards2 = np.zeros([3, 49])
# plot_rewards3 = np.zeros([3, 49])
# for j in range(49):
#     plot_rewards1[0, j] = np.mean(ddpg1[j * 100:(j + 1) * 100])
#     plot_rewards1[1, j] = np.mean(ddpg2[j * 100:(j + 1) * 100])
#     plot_rewards1[2, j] = np.mean(ddpg3[j * 100:(j + 1) * 100])
#     #
#     plot_rewards2[0, j] = np.mean(ppo1[j * 100:(j + 1) * 100])
#     plot_rewards2[1, j] = np.mean(ppo2[j * 100:(j + 1) * 100])
#     plot_rewards2[2, j] = np.mean(ppo3[j * 100:(j + 1) * 100])
#
#     plot_rewards3[0, j] = np.mean(naf1[j * 100:(j + 1) * 100])
#     plot_rewards3[1, j] = np.mean(naf2[j * 100:(j + 1) * 100])
#     plot_rewards3[2, j] = np.mean(naf3[j * 100:(j + 1) * 100])
#
# sns_plot1 = sns.tsplot(data=plot_rewards1,color='r')
# sns_plot2 = sns.tsplot(data=plot_rewards2,color='g')
# sns_plot3 = sns.tsplot(data=plot_rewards3,color='b')
#
# fig = sns_plot1.get_figure()
# plt.legend([sns_plot1,sns_plot2,sns_plot3],
#            bbox_to_anchor=(0, 1), loc=2, borderaxespad=0.,
#            labels=[r'DDPG',r'PPO',r'NAF'],
#            prop={'size':16})
#
# plt.xlabel('Episodes*100', fontsize=16)
# plt.ylabel('Reward', fontsize=16)
# plt.gca().tick_params(axis='x', which='both', labeltop='off', labelbottom='on', labelsize=16)
# plt.gca().tick_params(axis='y', which='both', labelsize=16)
#
# fig.savefig("figure.pdf", bbox_inches='tight')



# ############################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
#
# # data = np.load("hdrqn/datas/hdrqn_v2/2/new_state_reached_partial_1_meta_16_sub_8.npy")
# ddpg1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ddpg2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ddpg3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# ppo1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# ppo2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# ppo3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/ppo_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
# naf1 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# naf2 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# naf3 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/naf_fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
#
# N = 5
# ddpg = (ddpg1+ddpg2+ddpg3)/3.0
# ddpg_data = (np.mean(ddpg[0:1000]),
#              np.mean(ddpg[1001:2000]),
#              np.mean(ddpg[2001:3000]),
#              np.mean(ddpg[3001:4000]),
#              np.mean(ddpg[4001:5000]))
#
# ind = np.arange(N)  # the x locations for the groups
# width = 0.1       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, ddpg_data, width, color="red")
#
# ppo = (ppo1+ppo2+ppo3)/3.0
# ppo_data = (np.mean(ppo[0:1000]),
#              np.mean(ppo[1001:2000]),
#              np.mean(ppo[2001:3000]),
#              np.mean(ppo[3001:4000]),
#              np.mean(ppo[4001:5000]))
#
# rects2 = ax.bar(ind + width, ppo_data, width, color="green")
#
# naf = (naf1+naf2+naf3)/3.0
# naf_data = (np.mean(naf[0:1000]),
#              np.mean(naf[1001:2000]),
#              np.mean(naf[2001:3000]),
#              np.mean(naf[3001:4000]),
#              np.mean(naf[4001:5000]))
# rects3 = ax.bar(ind + width*2, naf_data, width, color="blue")
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Reward', fontsize=14)
# ax.set_xlabel('Episodes', fontsize=14)
# # ##################################################
# # # Hide major tick labels
# ax.set_xticklabels('')
# ax.set_xticks([0.25,1.25,2.25,3.25,4.25], minor=True)
# ax.set_xticklabels(['1000','2000','3000','4000','5000'], minor=True, fontsize=14)
# # # #######################################
# plt.gca().tick_params(axis='y', which='both', labelsize=14)
#
# # axbox = ax.get_position()
# # ax.set_position([axbox.x0, axbox.y0, axbox.width * 0.70, axbox.height])
#
# # ax.set_position([axbox.x0, axbox.y0 + axbox.height * 0.1,
# #                  axbox.width, axbox.height * 0.9])
# ax.legend((rects1[0], rects2[0], rects3[0]),
#           ('DDPG', 'PPO', "NAF"),
#             bbox_to_anchor=(0.72, 0.2), loc=2, borderaxespad=0.,
#           # loc='center left', bbox_to_anchor=(1, 0.5),
#           # loc=(axbox.x1 + 0.1, (axbox.y0+axbox.y1)/2.0-0.2),
#           prop={'size': 12})
# fig.savefig("compare_baseline.pdf", bbox_inches='tight')
# plt.show()


# # Bar plot
# ############################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
#
# d01 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_1/total_reward.npy")
# d02 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_2/total_reward.npy")
# d03 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_3/total_reward.npy")
# d21 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_1/total_reward.npy")
# d22 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_2/total_reward.npy")
# d23 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_3/total_reward.npy")
# d201 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/total_reward.npy")
# d202 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/total_reward.npy")
# d203 = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/total_reward.npy")
#
# N = 5
# d0 = (d01+d02+d03)/3.0
# ddpg_data = (np.mean(d0[0:1000]),
#              np.mean(d0[1001:2000]),
#              np.mean(d0[2001:3000]),
#              np.mean(d0[3001:4000]),
#              np.mean(d0[4001:5000]))
#
# ind = np.arange(N)  # the x locations for the groups
# width = 0.1       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, ddpg_data, width, color="red")
#
# d2 = (d21+d22+d23)/3.0
# ppo_data = (np.mean(d2[0:1000]),
#              np.mean(d2[1001:2000]),
#              np.mean(d2[2001:3000]),
#              np.mean(d2[3001:4000]),
#              np.mean(d2[4001:5000]))
#
# rects2 = ax.bar(ind + width, ppo_data, width, color="green")
#
# d20 = (d201+d202+d203)/3.0
# naf_data = (np.mean(d20[0:1000]),
#              np.mean(d20[1001:2000]),
#              np.mean(d20[2001:3000]),
#              np.mean(d20[3001:4000]),
#              np.mean(d20[4001:5000]))
# rects3 = ax.bar(ind + width*2, naf_data, width, color="blue")
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Reward', fontsize=14)
# ax.set_xlabel('Episodes', fontsize=14)
# # ##################################################
# # # Hide major tick labels
# ax.set_xticklabels('')
# ax.set_xticks([0.25,1.25,2.25,3.25,4.25], minor=True)
# ax.set_xticklabels(['1000','2000','3000','4000','5000'], minor=True, fontsize=14)
# # # #######################################
# plt.gca().tick_params(axis='y', which='both', labelsize=14)
#
# ax.legend((rects1[0], rects2[0], rects3[0]),
#           ("d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"),
#             bbox_to_anchor=(0.7, 0.2), loc=2, borderaxespad=0.,
#           # loc='center left', bbox_to_anchor=(1, 0.5),
#           # loc=(axbox.x1 + 0.1, (axbox.y0+axbox.y1)/2.0-0.2),
#           prop={'size': 12})
# fig.savefig("compare_d.pdf", bbox_inches='tight')
# plt.show()


# # success rate
# # ############################################################################################
# d01 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_1/back_trajectory.npy")[0])
# # d02 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_2/back_trajectory.npy")[0])
# # d03 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.0_trial_3/back_trajectory.npy")[0])
# d21 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_1/back_trajectory.npy")[0])
# # d22 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_2/back_trajectory.npy")[0])
# # d23 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_trial_3/back_trajectory.npy")[0])
# d201 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_1/back_trajectory.npy")[0])
# # d202 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_2/back_trajectory.npy")[0])
# # d203 = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.2_trial_3/back_trajectory.npy")[0])
#
# N = 5
# d0 = np.zeros(5000)
# d2 = np.zeros(5000)
# d20 = np.zeros(5000)
# for i in range(5000):
#     x = d01[i, 0][len(d01[i, 0])-1]
#     y = d01[i, 1][len(d01[i, 0])-1]
#     sqrd_dist_to_goal = (60.0 - x) ** 2 + (65.0 - y) ** 2
#     temp = np.sqrt(np.max([0, sqrd_dist_to_goal - 1.1**2]))
#     if (temp == 0.0):
#         d0[i] = 1.0
#     else:
#         d0[i] = 0.0
#
#     x = d21[i, 0][len(d21[i, 0]) - 1]
#     y = d21[i, 1][len(d21[i, 0]) - 1]
#     sqrd_dist_to_goal = (60.0 - x) ** 2 + (65.0 - y) ** 2
#     temp = np.sqrt(np.max([0, sqrd_dist_to_goal - 1.1**2]))
#     if (temp == 0.0):
#         d2[i] = 1.0
#     else:
#         d2[i] = 0.0
#
#     x = d201[i, 0][len(d201[i, 0]) - 1]
#     y = d201[i, 1][len(d201[i, 0]) - 1]
#     sqrd_dist_to_goal = (60.0 - x) ** 2 + (65.0 - y) ** 2
#     temp = np.sqrt(np.max([0, sqrd_dist_to_goal - 1.1**2]))
#     if (temp == 0.0):
#         d20[i] = 1.0
#     else:
#         d20[i] = 0.0
#
# d0_data = (np.mean(d0[0:1000]),
#              np.mean(d0[1001:2000]),
#              np.mean(d0[2001:3000]),
#              np.mean(d0[3001:4000]),
#              np.mean(d0[4001:5000]))
#
# ind = np.arange(N)  # the x locations for the groups
# width = 0.1       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, d0_data, width, color="red")
#
# d2_data = (np.mean(d2[0:1000]),
#              np.mean(d2[1001:2000]),
#              np.mean(d2[2001:3000]),
#              np.mean(d2[3001:4000]),
#              np.mean(d2[4001:5000]))
#
# rects2 = ax.bar(ind + width, d2_data, width, color="green")
#
# d20_data = (np.mean(d20[0:1000]),
#              np.mean(d20[1001:2000]),
#              np.mean(d20[2001:3000]),
#              np.mean(d20[3001:4000]),
#              np.mean(d20[4001:5000]))
# rects3 = ax.bar(ind + width*2, d20_data, width, color="blue")
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Success rate', fontsize=14)
# ax.set_xlabel('Episodes', fontsize=14)
# # ##################################################
# # # Hide major tick labels
# ax.set_xticklabels('')
# ax.set_xticks([0.25,1.25,2.25,3.25,4.25], minor=True)
# ax.set_xticklabels(['1000','2000','3000','4000','5000'], minor=True, fontsize=14)
# # # #######################################
# plt.gca().tick_params(axis='y', which='both', labelsize=14)
#
# ax.legend((rects1[0], rects2[0], rects3[0]),
#           ("d = 0 cm","d = [-2, 2] cm","d = [-20, 20] cm"),
#             bbox_to_anchor=(0.0, 0.95), loc=2, borderaxespad=0.,
#           # loc='center left', bbox_to_anchor=(1, 0.5),
#           # loc=(axbox.x1 + 0.1, (axbox.y0+axbox.y1)/2.0-0.2),
#           prop={'size': 12})
# fig.savefig("ratio_d.pdf", bbox_inches='tight')
# plt.show()


# # success rate dynamic velocity
# # ############################################################################################
# dynamic_back = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/back_trajectory.npy")[0])
# fixed_back = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/back_trajectory.npy")[0])
# dynamic_front = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_dynamic_velocity_d_0.02_trial_1/front_trajectory.npy")[0])
# fixed_front = np.asarray(np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/fix_goal_fix_velocity_d_0.02_opposite_goal_trial_1/front_trajectory.npy")[0])
#
# N = 5
# d1 = np.zeros(10000)
# d2 = np.zeros(10000)
# d1_opposite = np.zeros(5)
# d2_opposite = np.zeros(5)
# for i in range(10000):
#     x = fixed_front[i, 0][0]
#     y = fixed_front[i, 1][0]
#     if (x < 50 and y < 50):
#         x = fixed_back[i, 0][len(fixed_back[i, 0])-1]
#         y = fixed_back[i, 1][len(fixed_back[i, 0])-1]
#         sqrd_dist_to_goal = (60.0 - x) ** 2 + (65.0 - y) ** 2
#         temp = np.sqrt(np.max([0, sqrd_dist_to_goal - 16.6**2]))
#         if (temp == 0.0):
#             d1[i] = 1.0
#         else:
#             d1[i] = 0.0
#
#         if (i < 2000):
#             d1_opposite[0] += 1
#         elif (i < 4000):
#             d1_opposite[1] += 1
#         elif (i < 6000):
#             d1_opposite[2] += 1
#         elif (i < 8000):
#             d1_opposite[3] += 1
#         elif (i < 10000):
#             d1_opposite[4] += 1
#
#     x = dynamic_front[i, 0][0]
#     y = dynamic_front[i, 1][0]
#     if (x < 50 and y < 50):
#         x = dynamic_back[i, 0][len(dynamic_back[i, 0]) - 1]
#         y = dynamic_back[i, 1][len(dynamic_back[i, 0]) - 1]
#         sqrd_dist_to_goal = (60.0 - x) ** 2 + (65.0 - y) ** 2
#         temp = np.sqrt(np.max([0, sqrd_dist_to_goal - 16.6**2]))
#         if (temp == 0.0):
#             d2[i] = 1.0
#         else:
#             d2[i] = 0.0
#
#         if (i < 2000):
#             d2_opposite[0] += 1
#         elif (i < 4000):
#             d2_opposite[1] += 1
#         elif (i < 6000):
#             d2_opposite[2] += 1
#         elif (i < 8000):
#             d2_opposite[3] += 1
#         elif (i < 10000):
#             d2_opposite[4] += 1
#
# d0_data = (np.sum(d1[0:2000])/d1_opposite[0],
#              np.sum(d1[2001:4000])/d1_opposite[1],
#              np.sum(d1[4001:6000])/d1_opposite[2],
#              np.sum(d1[6001:8000])/d1_opposite[3],
#              np.sum(d1[8001:10000])/d1_opposite[4])
#
# ind = np.arange(N)  # the x locations for the groups
# width = 0.1       # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(ind, d0_data, width, color="red")
#
# d2_data = (np.sum(d2[0:2000])/d2_opposite[0],
#              np.sum(d2[2001:4000])/d2_opposite[1],
#              np.sum(d2[4001:6000])/d2_opposite[2],
#              np.sum(d2[6001:8000])/d2_opposite[3],
#              np.sum(d2[8001:10000])/d2_opposite[4])
#
# rects2 = ax.bar(ind + width, d2_data, width, color="blue")
#
# # add some text for labels, title and axes ticks
# ax.set_ylabel('Success rate to turn', fontsize=14)
# ax.set_xlabel('Episodes', fontsize=14)
# # ##################################################
# # # Hide major tick labels
# ax.set_xticklabels('')
# ax.set_xticks([0.25,1.25,2.25,3.25,4.25], minor=True)
# ax.set_xticklabels(['2000','4000','6000','8000','10000'], minor=True, fontsize=14)
# # # #######################################
# plt.gca().tick_params(axis='y', which='both', labelsize=14)
#
# ax.legend((rects1[0], rects2[0]),
#           ("fixed velocity","dynamic velocity"),
#             bbox_to_anchor=(0.0, 0.95), loc=2, borderaxespad=0.,
#           # loc='center left', bbox_to_anchor=(1, 0.5),
#           # loc=(axbox.x1 + 0.1, (axbox.y0+axbox.y1)/2.0-0.2),
#           prop={'size': 12})
# fig.savefig("ratio_dynamic_vs_fixed.pdf", bbox_inches='tight')
# plt.show()


# report states
#################################################################################
import numpy as np
import matplotlib.pyplot as plt
states = np.load("/home/tuyen/syoon/deeprl/datas/DDPG/Bicycle-v1/train/journal/states/printed_states.npy")

selected_trajectory = None
for i in range(4500,5000):
    if (len(states[i])<400):
        selected_trajectory = np.asarray(states[i])
        break

x = np.linspace(0, 100,100)

y1 = selected_trajectory[0:100,0]
y2 = selected_trajectory[0:100,1]
y3 = selected_trajectory[0:100,2]
y4 = selected_trajectory[0:100,3]
y5 = selected_trajectory[0:100,4]
y6 = selected_trajectory[0:100,5]

fig, ax = plt.subplots()

plt.subplot(6, 1, 1)
plt.plot(x, y1)
plt.ylabel(r'$\theta$')

plt.subplot(6, 1, 2)
plt.plot(x, y2)
plt.ylabel(r'$\dot{\theta}$')

plt.subplot(6, 1, 3)
plt.plot(x, y3)
plt.ylabel(r'$\omega$')

plt.subplot(6, 1, 4)
plt.plot(x, y4)
plt.ylabel(r'$\dot{\omega}$')

plt.subplot(6, 1, 5)
plt.plot(x, y5)
plt.ylabel(r'$\ddot{\omega}$')

plt.subplot(6, 1, 6)
plt.plot(x, y6)
plt.ylabel(r'$\psi_g$')
plt.xlabel('timesteps (1 timesteps = 0.025ms)')

fig.savefig("printed_states.pdf", bbox_inches='tight')
plt.show()