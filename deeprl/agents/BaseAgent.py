import tensorflow as tf
import os
import numpy as np
from gym.wrappers import Monitor
from gym.wrappers.time_limit import TimeLimit


class BaseAgent(object):
    def __init__(self, sess, env, env_name, max_episode,
                 warm_up=5000, max_test_epoch=10,save_plot_interval=100,
                 render=True, save_interval=10, max_step_per_game = 10000):
        """
        Base agent. Provide basic functions: save, restore, perform training and evaluation (abstract method).
        Args:
            sess: tf.Session() variable
            memory: Replay Memory
            env: OpenAI Gym environment or a wrapped environment
            env_name: a string indicating the name of the env
            max_episode: maximum step to perform training
            evaluate_every: how many episode to evaluate the current policy
            warm_up: how many steps to take on random policy
            max_test_epoch: maximum epochs to evaluate the policy after finishing training
            render: if show the training process
        """
        self.sess = sess
        self.max_step_per_game = max_step_per_game
        self.max_test_epoch = max_test_epoch
        self.env_name = env_name
        self.warm_up = warm_up
        self.monitor_dir = os.path.join("tmp", type(self).__name__, env_name)

        # to be wrapped during evaluation
        self.env = env

        # for summary writer
        self.logdir = os.path.join("logs", type(self).__name__, env_name)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        # for plot data
        self.datadir = os.path.join("datas", type(self).__name__, env_name)
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)

        # for plot data
        self.figdir = os.path.join("figures", type(self).__name__, env_name)
        if not os.path.exists(self.figdir):
            os.makedirs(self.figdir)

        # Set up summary Ops
        self.summary_ops, self.summary_vars = self.build_summaries()
        self.writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

        self.render = render
        self.gamma = 0.99
        self.max_episode = max_episode
        self.save_interval = save_interval
        self.save_plot_interval = save_plot_interval

        # save directory
        self.save_dir = os.path.join("models", env_name)
        # check if the save directory is there
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # create saver
        self.saver = tf.train.Saver()

    def save_plot_data(self,filename, data, is_train=True):
        if (is_train == True):
            path = os.path.join(self.datadir, "train")
        else:
            path = os.path.join(self.datadir, "evaluate")

        if not os.path.exists(path):
            os.makedirs(path)

        np.save(os.path.join(path, filename), data)
        print("Saving the plot data to path %s" % path)

    def restore_plot_data(self, filename, is_train=True):
        if (is_train == True):
            path = os.path.join(self.datadir, "train", filename)
        else:
            path = os.path.join(self.datadir, "evaluate", filename)

        data = np.load(path)
        print("Successfully load the data at path %s" % path)
        return data

    def save_plot_figure(self, plt, filename):
        path = os.path.join(self.figdir, filename)
        plt.savefig(path)
        print("Saving the figure to path %s" % path)

    def save(self):
        """
        Save all model parameters and replay memory to self.save_dir folder.
        The save_path should be models/env_name/name_of_agent.
        """
        # path to the checkpoint name
        path = os.path.join(self.save_dir, type(self).__name__)
        print("Saving the model to path %s" % path)
        self.saver.save(self.sess, path)
        print("Done saving!")

    def restore(self):
        """
        Restore model parameters and replay memory from self.save_dir folder.
        The name of the folder should be models/env_name
        """
        ckpts = tf.train.get_checkpoint_state(self.save_dir)
        if ckpts and ckpts.model_checkpoint_path:
            ckpt = ckpts.model_checkpoint_path
            self.saver.restore(self.sess, ckpt)
            print("Successfully load the model %s" % ckpt)
        else:
            print("Model Restore Failed %s" % self.save_dir)

    # ===========================
    #   Tensorflow Summary Ops
    # ===========================
    def build_summaries(self):
        episode_reward = tf.Variable(0.)
        tf.summary.scalar('Rewards', episode_reward)
        episode_ave_max_q = tf.Variable(0.)
        tf.summary.scalar('Q-values', episode_ave_max_q)
        episode_step = tf.Variable(0.)
        tf.summary.scalar('Steps', episode_step)

        summary_vars = [episode_reward, episode_ave_max_q, episode_step]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars

    def train(self):
        """
        Train the model. The agent training process will end at self.max_step.
        If max_step_per_game is provided, the agent will perform a limited steps
        during each game.
        """
        raise NotImplementedError("This method should be implemented")

    def evaluate(self):
        """
        Evaluate the model. This should only be called when self.max_epoch is reached.
        The evaluation will be recorded.
        """
        raise NotImplementedError("This method should be implemented")

