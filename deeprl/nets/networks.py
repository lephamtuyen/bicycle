from __future__ import division
from __future__ import print_function
from  __future__ import absolute_import
from nets.nn_ops import *
import tensorflow as tf

# Soft target update param
TAU = 0.001
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001

class BaseNetwork(object):
    def __init__(self, state_dim, action_dim, name, initializer=tf.contrib.layers.xavier_initializer()):
        """
        Abstarct class for creating networks
        :param state_dim:
        :param action_dim:
        :param stddev:
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initializer = initializer

        # build network
        self.network = self.build(name)
        self.network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                              "target" not in v.name]

        # build target
        self.target_network = self.build("target_%s" % name)
        self.target_network_param = [v for v in tf.get_collection(tf.GraphKeys.MODEL_VARIABLES) if name in v.name and
                             "target" in v.name]

        self.gradients = None

        # optimizer
        self.optimizer = self.create_optimizer()

    def create_update_op(self):
        update_op = [tf.assign(target_network_param, (1 - TAU) * target_network_param + TAU * network_param)
                        for target_network_param, network_param in zip(self.target_network_param, self.network_param)]

        return update_op

    def create_train_op(self):
        return self.optimizer.apply_gradients([(g, v) for g, v in zip(self.gradients, self.network_param)])

    def build(self, name):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def create_optimizer(self):
        """
        Abstract method, to be implemented by child classes
        """
        raise NotImplementedError("Not implemented")

    def compute_gradient(self):
        """
        Abstract method, compute gradient in order to be used by self.optimizer
        """
        raise NotImplementedError("Not implemented")


class CriticNetwork(BaseNetwork):
    def __init__(self, state_dim, action_dim, name="critic"):
        """
        Initialize critic network. The critic network maintains a copy of itself and target updating ops
        Args
            state_dim: dimension of input space, if is length one, we assume it is low dimension.
            action_dim: dimension of action space.
        """
        super(CriticNetwork, self).__init__(state_dim, action_dim, name=name)

        self.update_op = self.create_update_op()

        # online critic
        self.network, self.state, self.action = self.network

        #target critic
        self.target_network, self.target_state, self.target_action = self.target_network

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")
        self.mean_loss = tf.reduce_mean(tf.squared_difference(self.y, self.network))
        self.loss = tf.squared_difference(self.y, self.network)

        # get gradients
        self.gradients = self.compute_gradient()

        # get action gradients
        self.action_gradient = self.compute_action_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(CRITIC_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        action = tf.placeholder(tf.float32, shape=[None, self.action_dim], name="%s_action" % name)
        with tf.variable_scope(name):
            net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
            net = dense_layer(tf.concat((net, action),1), 300, use_bias=True, scope="fc2",
                                  initializer=self.initializer)
            net = tf.nn.relu(net)

            # for low dim, weights are from uniform[-3e-3, 3e-3]
            net = dense_layer(net, 1, initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="q",
                                  use_bias=True)
        return tf.squeeze(net), x, action

    def compute_gradient(self):
        grad = tf.gradients(self.mean_loss, self.network_param, name="critic_gradients")
        return grad

    def compute_action_gradient(self):
        action_gradient = tf.gradients(self.network, self.action, name="action_gradients")
        return action_gradient


class ActorNetwork(BaseNetwork):
    def __init__(self, state_dim, action_dim, name="actor"):
        """
        Initialize actor network
        """
        super(ActorNetwork, self).__init__(state_dim, action_dim, name=name)

        self.update_op = self.create_update_op()

        # online actor
        self.network, self.state = self.network

        # target actor
        self.target_network, self.target_state = self.target_network

        # for actor network, we need to know the action gradient in critic network
        self.action_gradient = tf.placeholder(tf.float32, shape=(None, action_dim), name="action_gradients")
        self.gradients = self.compute_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(ACTOR_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        with tf.variable_scope(name):
            net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
            net = tf.nn.relu(dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer))

            # use tanh to normalize output between [-1, 1]
            net = tf.nn.tanh(dense_layer(net, self.action_dim,
                                             initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                             scope="pi", use_bias=True))

            return net, x

    def compute_gradient(self):
        grads = tf.gradients(self.network, self.network_param, -self.action_gradient)
        return grads

class ActorController(BaseNetwork):
    def __init__(self, sess, state_dim, action_dim, name="actor_controller"):
        """
        Initialize actor network
        """
        super(ActorController, self).__init__(state_dim, action_dim, name=name)

        self.sess = sess

        self.update_op = self.create_update_op()
        self.network, self.action, self.maxq, self.state, self.goal = self.network
        self.target_network, self.target_action, self.target_maxq, self.target_state, self.target_goal = self.target_network

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")

        # current Q
        self.actions = tf.placeholder(tf.int32, shape=[None], name='action')
        self.actions_one_hot = tf.one_hot(self.actions, self.action_dim, name='action_one_hot')
        self.cur_q = tf.reduce_sum(self.network * self.actions_one_hot, reduction_indices=1,
                                    name='current_q')

        self.mean_loss = tf.reduce_mean(tf.squared_difference(self.y, self.cur_q))

        # get gradients
        self.gradients = self.compute_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)

    def build(self, name):
        if "target" not in name:
            trainable = True
        else:
            trainable = False

        state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="%s_state" % name)
        goal = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="%s_goal" % name)
        input = tf.concat([state, goal],1)
        with tf.variable_scope(name):

            net = tf.nn.relu(dense_layer(input, 30, use_bias=True, scope="fc1", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, 30, use_bias=True, scope="fc2", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, 30, use_bias=True, scope="fc3", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, self.action_dim, use_bias=True, scope="out", initializer=self.initializer, trainable=trainable))

            max_q = tf.reduce_max(net, reduction_indices=1)
            action = tf.argmax(net, dimension=1)
        return net, action, max_q, state, goal

    def compute_gradient(self):
        grad = tf.gradients(self.mean_loss, self.network_param, name="critic_gradients")
        return grad

    def create_update_op(self):
        update_op = [tf.assign(target_network_param, network_param)
                        for target_network_param, network_param in zip(self.target_network_param, self.network_param)]

        return update_op

    def actor_predict(self, state, goal):
        return self.sess.run(self.action,
                             feed_dict={self.state: state,
                                        self.goal: goal})

    def get_maxq(self, state, goal):
        return self.sess.run(self.target_maxq,
                             feed_dict={self.target_state: state,
                                        self.target_goal: goal})

    def update_actor_params(self, y, state, action, goal):
        return self.sess.run([self.mean_loss, self.train],
                             feed_dict={self.y: y,
                                        self.state: state,
                                        self.actions: action,
                                        self.goal: goal})

class MetaController(BaseNetwork):
    def __init__(self, sess, state_dim, name="meta_controller"):
        """
        Initialize actor network
        """
        super(MetaController, self).__init__(state_dim, state_dim, name=name)

        self.sess = sess
        self.update_op = self.create_update_op()
        self.network, self.goal, self.maxq, self.state = self.network
        self.target_network, self.target_goal, self.target_maxq, self.target_state = self.target_network

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, action(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")

        # current Q
        self.goals = tf.placeholder(tf.int32, shape=[None], name='action')
        self.goal_one_hot = tf.one_hot(self.goals, self.action_dim, 1.0, 0.0, name='goal_one_hot')
        self.cur_q = tf.reduce_sum(self.network * self.goal_one_hot, reduction_indices=1,
                                    name='current_q')

        self.mean_loss = tf.reduce_mean(tf.squared_difference(self.y, self.cur_q))

        # get gradients
        self.gradients = self.compute_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.RMSPropOptimizer(0.00025, momentum=0.95, epsilon=0.01)

    def build(self, name):
        if "target" not in name:
            trainable = True
        else:
            trainable = False

        state = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name="%s_state" % name)
        with tf.variable_scope(name):

            net = tf.nn.relu(dense_layer(state, 30, use_bias=True, scope="fc1", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, 30, use_bias=True, scope="fc2", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, 30, use_bias=True, scope="fc3", initializer=self.initializer, trainable=trainable))
            net = tf.nn.relu(dense_layer(net, self.state_dim, use_bias=True, scope="out", initializer=self.initializer, trainable=trainable))

            goal = tf.argmax(net, dimension=1)
            max_q = tf.reduce_max(net, reduction_indices=1)
        return net, goal, max_q, state

    def compute_gradient(self):
        grad = tf.gradients(self.mean_loss, self.network_param, name="critic_gradients")
        return grad

    def create_update_op(self):
        update_op = [tf.assign(target_network_param, network_param)
                        for target_network_param, network_param in zip(self.target_network_param, self.network_param)]

        return update_op

    def meta_predict(self, state):
        return self.sess.run(self.goal,
                             feed_dict={self.state: state})

    def get_maxq(self, state):
        return self.sess.run(self.target_maxq,
                             feed_dict={self.target_state: state})

    def update_meta_params(self, y, state, goal):
        return self.sess.run([self.mean_loss, self.train],
                             feed_dict={self.y: y,
                                        self.state: state,
                                        self.goals: goal})


class MetaCriticNetwork(BaseNetwork):
    def __init__(self, state_dim, subgoal_dim, name="meta_critic"):
        self.subgoal_dim = subgoal_dim

        super(MetaCriticNetwork, self).__init__(state_dim, subgoal_dim, name=name)

        self.update_op = self.create_update_op()
        self.network, self.state, self.subgoal = self.network
        self.target_network, self.target_state, self.target_subgoal = self.target_network

        # for critic network, the we need one more input variable: y to compute the loss
        # this input variable is fed by: r + gamma * target(s_t+1, subgoal(s_t+1))
        self.y = tf.placeholder(tf.float32, shape=None, name="target_q")
        self.mean_loss = tf.reduce_mean(tf.squared_difference(self.y, self.network))

        # get gradients
        self.gradients = self.compute_gradient()

        # get subgoal gradients
        self.subgoal_gradient = self.compute_subgoal_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(CRITIC_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        subgoal = tf.placeholder(tf.float32, shape=[None, self.subgoal_dim], name="%s_subgoal" % name)
        with tf.variable_scope(name):
            if len(self.state_dim) == 1:
                net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                net = dense_layer(tf.concat((net, subgoal),1), 300, use_bias=True, scope="fc2",
                                  initializer=self.initializer)
                net = tf.nn.relu(net)

                # for low dim, weights are from uniform[-3e-3, 3e-3]
                net = dense_layer(net, 1, initializer=tf.random_uniform_initializer(-3e-3, 3e-3), scope="q",
                                  use_bias=True)
            else:
                # first convolutional layer with stride 4
                net = conv2d(x, 3, initializer=self.initializer, output_size=32, scope="conv1", stride=4, use_bias=True)
                net = tf.nn.relu(net)

                # second convolutional layer with stride 1
                net = conv2d(net, 3, stride=2, output_size=32, initializer=self.initializer, scope="conv2",
                             use_bias=True)
                net = tf.nn.relu(net)

                # third convolutional layer with stride 1
                net = conv2d(net, 3, stride=1, output_size=32, initializer=self.initializer, use_bias=True,
                             scope="conv3")
                net = tf.nn.relu(net)

                # first dense layer
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer, scope="fc1",
                                             use_bias=True))

                # second dense layer with subgoal embedded
                net = tf.nn.relu(dense_layer(tf.concat((net, subgoal),1), output_dim=200, initializer=self.initializer,
                                             scope="fc2", use_bias=True))

                # Q layer
                net = dense_layer(net, output_dim=1, initializer=tf.random_uniform_initializer(-4e-4, 4e-4), scope="Q",
                                  use_bias=True)
        return tf.squeeze(net), x, subgoal

    def compute_gradient(self):
        grad = tf.gradients(self.mean_loss, self.network_param, name="critic_gradients")
        return grad

    def compute_subgoal_gradient(self):
        subgoal_gradient = tf.gradients(self.network, self.subgoal, name="subgoal_gradients")
        return subgoal_gradient


class MetaActorNetwork(BaseNetwork):
    def __init__(self, state_dim, subgoal_dim, name="actor"):

        self.subgoal_dim = subgoal_dim
        super(MetaActorNetwork, self).__init__(state_dim, subgoal_dim, name=name)

        self.update_op = self.create_update_op()
        self.network, self.state = self.network
        self.target_network, self.target_state = self.target_network

        # for actor network, we need to know the subgoal gradient in critic network
        self.subgoal_gradient = tf.placeholder(tf.float32, shape=(None, subgoal_dim), name="subgoal_gradients")
        self.gradients = self.compute_gradient()

        self.train = self.create_train_op()

    def create_optimizer(self):
        return tf.train.AdamOptimizer(ACTOR_LEARNING_RATE)

    def build(self, name):
        x = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.state_dim, name="%s_input" % name)
        with tf.variable_scope(name):
            if len(self.state_dim) == 1:
                net = tf.nn.relu(dense_layer(x, 400, use_bias=True, scope="fc1", initializer=self.initializer))
                net = tf.nn.relu(dense_layer(net, 300, use_bias=True, scope="fc2", initializer=self.initializer))
                # use tanh to normalize output between [-1, 1]
                net = tf.nn.tanh(dense_layer(net, self.subgoal_dim,
                                             initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                                             scope="pi", use_bias=True))
            else:
                # first convolutional layer with stride 4
                net = conv2d(x, 3, stride=4, output_size=32, initializer=self.initializer, scope="conv1", use_bias=True)
                net = tf.nn.relu(net)

                # second convolutional layer with stride 2
                net = conv2d(net, 3, stride=2, output_size=32, initializer=self.initializer, scope="conv2", use_bias=True)
                net = tf.nn.relu(net)

                # third convolutional layer with stride 1
                net = conv2d(net, 3, stride=1, output_size=32, initializer=self.initializer, scope="conv3", use_bias=True)
                net = tf.nn.relu(net)

                # first dense layer
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer, scope="fc1", use_bias=True))

                # second dense layer with subgoal embedded
                net = tf.nn.relu(dense_layer(net, output_dim=200, initializer=self.initializer, scope="fc2", use_bias=True))
                # Q layer
                net = tf.tanh(dense_layer(net, output_dim=self.state_dim,
                                          initializer=tf.random_uniform_initializer(-4e-4, 4e-4),
                                          scope="pi", use_bias=True))
            return net, x

    def compute_gradient(self):
        grads = tf.gradients(self.network, self.network_param, -self.subgoal_gradient)
        return grads