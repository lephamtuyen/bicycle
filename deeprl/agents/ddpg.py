from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from agents.BaseAgent import BaseAgent
from utils.utilities import *
import os
from noises.OUNoise import *
from gym.wrappers import Monitor
from gym.spaces import *
from envs.env_wrapper import *
from mems.replay import *
from nets.networks import *

class DDPG(BaseAgent):
    def __init__(self, sess, actor, critic, memory, env, env_name, max_episode=3000, warm_up=5000,is_plot=None,
                 max_test_epoch=10, render=True, save_interval=10,save_plot_interval=100,
                 noise_theta = 0.15, noise_sigma = 0.2, max_step_per_game = 10000):
        """
        Args:
            sess: tensorflow session
            actor: Actor networks
            critic: critic networks
            memory: replay memory
            env: gym ai environment
            env_name: the name of environment
            max_episode: the maximum number of episode which is used to training the agent
            warm_up: number of step the agent takes an random action before starting to training
            is_plot: If True, then store the trajectoreis
            max_test_epoch: the maximum number of episodes for evaluate the agent
            render: simulate the environment or not
            save_interval: save the policy for each save_interval episode
            save_plot_interval: save data to file every save_plot_interval episode
            noise_sigma and noise_theta: parameters of OU noise
            max_step_per_game: maximum number of step per game
        A deep deterministic policy gradient agent.
        """

        # Call parent function (Eg. BaseAgent)
        super(DDPG, self).__init__(sess, env=env, render=render, max_episode=max_episode,
                                   env_name=env_name, warm_up=warm_up,save_plot_interval=save_plot_interval,
                                   max_test_epoch=max_test_epoch, save_interval=save_interval, max_step_per_game = max_step_per_game)

        # Assign parameter to the variables of the class
        self.critic = critic
        self.actor = actor
        self.memory = memory
        self.batch_size = self.memory.batch_size
        self.is_plot = is_plot

        # Initialize the exploration noise
        self.noise = OUNoise(self.actor.action_dim, theta=noise_theta, sigma=noise_sigma)

        np.random.seed(123)
        tf.set_random_seed(123)
        if type(self.env) == ContinuousWrapper:
            self.env.seed(123)

        # Initialize the whole network policy
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        print("Start training....")

        episodes = 0

        # Initialize the variables for saving the trajectories
        if self.is_plot:
            total_reward = []
            total_step = []
            if "Bicycle" in self.env_name:
                back_trajectory = []
                front_trajectory = []
                term1 = []
                term2 = []
                term3 = []
                term4 = []
                velocity = []
                printed_states = []
                # goals = []

        # iterate until reach the maximum step
        while episodes < self.max_episode:

            # re-initialize per episode variables
            current_state = self.env.reset()
            current_state = current_state[np.newaxis]
            per_game_step = 0
            per_game_reward = 0
            per_game_q = 0
            done = False

            # re-initialize the noise process
            self.noise.reset()

            # iterate through the whole episode
            while not done and per_game_step < self.max_step_per_game:
                # simulate the environment
                if self.render:
                    self.env.render()

                # Take a random action in the warm-up stage
                if self.warm_up > self.memory.get_curr_size():
                    # take a random action to fill up the memory
                    action = self.env.action_space.sample()
                else:
                    # take a action that is generated from the actor
                    #noise = ((self.max_episode-episodes)/self.max_episode)*self.noise.noise()
                    noise = self.noise.noise()
                    pure_action = self.action(current_state)
                    action = pure_action + noise

                    action = action.flatten()

                # evaluate the q value
                reshaped_action = action.reshape(1, -1)
                per_game_q += self.sess.run(self.critic.network,
                                            feed_dict={self.critic.action: reshaped_action, self.critic.state: current_state})

                # Get next state
                next_state, reward, done, info = self.env.step(action)

                next_state = next_state[np.newaxis]
                terminal = 0 if done else 1

                # store experiences
                self.memory.add([current_state, next_state, reward, terminal, action], reward)

                # train the networks
                if self.warm_up < self.memory.get_curr_size():
                    if self.batch_size < self.memory.get_curr_size():
                        # random batch_size samples
                        samples, idxs = self.memory.sample()
                        s = samples[0]
                        next_s = samples[1]
                        r = samples[2]
                        t = samples[3]
                        a = samples[4]
                        a = a.reshape(self.batch_size, -1)

                        # step 1: target action
                        target_action = self.actor_target_predict(next_s)

                        # step 2: estimate next state's Q value according to this action (target)
                        y = self.critic_target_predict(next_s, target_action)
                        y = r + t * self.gamma * y

                        # step 3: update critic
                        _, loss, _ = self.update_critic(y, a, s)

                        # step 4: perceive action according to actor given s
                        actor_action = self.action(s)

                        # step 5: calculate action gradient
                        a_grads = self.get_action_gradient(s, actor_action)

                        # step 6: update actor policy
                        self.update_actor(s, a_grads[0])

                        # update target networks
                        self.sess.run([self.actor.update_op, self.critic.update_op])
                        # end of training

                current_state = next_state
                per_game_step += 1

                per_game_reward += reward

            if self.is_plot:
                total_reward.append(per_game_reward)
                total_step.append(per_game_step)
                if "Bicycle" in self.env_name:
                    back_trajectory.append([self.env.env.get_xbhist(), self.env.env.get_ybhist()])
                    front_trajectory.append([self.env.env.get_xfhist(), self.env.env.get_yfhist()])
                    terms = self.env.env.getRewardTerm()
                    term1.append(terms[0])
                    term2.append(terms[1])
                    term3.append(terms[2])
                    term4.append(terms[3])
                    velocity.append(self.env.env.get_vhist())
                    printed_states.append(self.env.env.get_shist())
                    # goals.append(self.env.env.get_goal())

            # if "Bicycle-v0" in self.env_name and per_game_step==self.max_step_per_game:
            #     back_trajectory.append([self.env.env.get_xbhist(), self.env.env.get_ybhist()])
            #     front_trajectory.append([self.env.env.get_xfhist(), self.env.env.get_yfhist()])
            #     self.save()
            #     break
            #
            # if "Bicycle-v1" in self.env_name and info["reach_goal"]==True:
            #     back_trajectory.append([self.env.env.get_xbhist(), self.env.env.get_ybhist()])
            #     front_trajectory.append([self.env.env.get_xfhist(), self.env.env.get_yfhist()])
            #     self.save()
            #     break

            # Save model
            if episodes % self.save_interval == 0 and episodes != 0:
                self.save()

            if self.is_plot and episodes % self.save_plot_interval == 0:
                self.save_plot_data("total_step", np.asarray(total_step), is_train=True)
                self.save_plot_data("total_reward", np.asarray(total_reward), is_train=True)
                if "Bicycle" in self.env_name:
                    if "Bicycle-v0" in self.env_name:
                        self.save_plot_data("back_trajectory", np.asarray([back_trajectory, [None, 1]]),
                                            is_train=True)
                        self.save_plot_data("front_trajectory", np.asarray([front_trajectory, [None, 1]]),
                                            is_train=True)
                    else:
                        self.save_plot_data("back_trajectory", np.asarray([back_trajectory, [np.array([self.env.env.x_goal,self.env.env.y_goal]), 1]]), is_train=True)
                        self.save_plot_data("front_trajectory", np.asarray([front_trajectory, [np.array([self.env.env.x_goal,self.env.env.y_goal]), 1]]), is_train=True)
                        self.save_plot_data("term1", np.asarray(term1), is_train=True)
                        self.save_plot_data("term2", np.asarray(term2), is_train=True)
                        self.save_plot_data("term3", np.asarray(term3), is_train=True)
                        self.save_plot_data("term4", np.asarray(term4), is_train=True)
                        self.save_plot_data("velocity", np.asarray(velocity), is_train=True)
                        self.save_plot_data("printed_states", np.asarray(printed_states), is_train=True)
                        # self.save_plot_data("goal", np.asarray(goals), is_train=True)

            episodes += 1
            per_game_q = per_game_q/per_game_step
            ################################################################
            summary_str = self.sess.run(self.summary_ops, feed_dict={
                self.summary_vars[0]: per_game_reward,
                self.summary_vars[1]: per_game_q,
                self.summary_vars[2]: per_game_step
            })

            self.writer.add_summary(summary_str, global_step=episodes)
            self.writer.flush()

            print("Episode: %s | Reward: %s | Q-value: %s | Steps: %s" % (episodes,
                                                                          per_game_reward,
                                                                          per_game_q,
                                                                          per_game_step))
            #################################################################

        if self.is_plot:
            self.save_plot_data("total_step", np.asarray(total_step),is_train=True)
            self.save_plot_data("total_reward", np.asarray(total_reward),is_train=True)
            if "Bicycle" in self.env_name:
                if "Bicycle-v0" in self.env_name:
                    self.save_plot_data("back_trajectory",
                                        np.asarray([back_trajectory, [None, 1]]),
                                        is_train=True)
                    self.save_plot_data("front_trajectory",
                                        np.asarray([front_trajectory, [None, 1]]),
                                        is_train=True)
                else:
                    self.save_plot_data("back_trajectory", np.asarray(
                        [back_trajectory,
                         [np.array([self.env.env.x_goal, self.env.env.y_goal]), 1]]),
                                        is_train=True)
                    self.save_plot_data("front_trajectory", np.asarray(
                        [front_trajectory,
                         [np.array([self.env.env.x_goal, self.env.env.y_goal]), 1]]),
                                        is_train=True)
                    self.save_plot_data("term1", np.asarray(term1), is_train=True)
                    self.save_plot_data("term2", np.asarray(term2), is_train=True)
                    self.save_plot_data("term3", np.asarray(term3), is_train=True)
                    self.save_plot_data("term4", np.asarray(term4), is_train=True)
                    self.save_plot_data("velocity", np.asarray(velocity), is_train=True)
                    self.save_plot_data("printed_states", np.asarray(printed_states), is_train=True)
                    # self.save_plot_data("goal", np.asarray(goals), is_train=True)

    def restoreModel(self):
        self.restore()

    def evaluate(self):
        print("Start Evaluating...")

        total_reward = 0

        # Restore the policy from file
        self.restore()

        # Iterate through the test episodes
        for epoch in range(self.max_test_epoch):
            print("Episode: %s" % (epoch))
            # start the environment
            state = self.env.reset()
            step = 0
            done = False
            total_reward = 0

            # start one episode
            while not done and step < self.max_step_per_game:
                if (self.render):
                    self.env.render()

                reshaped_state = state[np.newaxis]

                # predict action using the network policy
                action = self.action(reshaped_state)

                # get the next state, reward and terminate signal
                state, reward, done, _ = self.env.step(action.flatten())
                total_reward += reward
                step += 1

            print("Episode: %s | Reward: %s | Steps: %s" % (epoch,
                                                            total_reward,
                                                            step))



    def critic_target_predict(self, state, action):
        return self.sess.run(self.critic.target_network,
                             feed_dict={self.critic.target_action: action, self.critic.target_state: state})

    def action(self, state):
        return (self.sess.run(self.actor.network,
                              feed_dict={self.actor.state: state}))

    def actor_target_predict(self, state):
        return self.sess.run(self.actor.target_network,
                             feed_dict={self.actor.target_state: state})

    def update_critic(self, y, a, s):
        return self.sess.run([self.critic.mean_loss, self.critic.loss, self.critic.train],
                             feed_dict={self.critic.y: y, self.critic.action: a, self.critic.state: s})

    def update_actor(self, s, a_grads):
        self.sess.run(self.actor.train,
                      feed_dict= {self.actor.state: s, self.actor.action_gradient: a_grads})

    def get_action_gradient(self, s, a):
        return self.sess.run(self.critic.action_gradient,
                             feed_dict= {self.critic.state: s, self.critic.action: a})
