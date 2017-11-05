import numpy as np
import tensorflow as tf
from pathlib import Path

# reproducible
#np.random.seed(1)
#tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.INFO)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.009,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.ep_prices, self.ep_last_weights, self.ep_fp = [], [], []

        tf.reset_default_graph()

        self._build_net()

        self.sess = tf.Session()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        my_file = Path("logs/model.ckpt")
        if my_file.is_file():
            # Restore variables from disk.
            self.saver.restore(self.sess, "logs/model.ckpt")

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features, self.n_actions, 50], name="observations")
            #From NCHW to NHWC
            #x = tf.reshape(self.tf_obs, [1, 2, 3, 4])  # input
            #x = tf.reshape(self.tf_obs, [-1, self.n_features, self.n_actions, 50])  # input
            self.prices = tf.transpose(self.tf_obs, [0, 2, 3, 1])

            self.tf_vt = tf.placeholder(tf.float32, [None, self.n_actions, 1, 1], name="future_prices")
            self.tf_weights = tf.placeholder(tf.float32, [None, self.n_actions, 1, 20], name="historic_weights")

        self.conv1 = tf.layers.conv2d(
            inputs=self.prices,
            filters=2,
            kernel_size=[1, 3],
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            name='conv1',
        )
        self.conv2 = tf.layers.conv2d(
            inputs=self.conv1,
            filters=1,
            kernel_size=[1, 48],
            activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name='conv2',
        )
        concat = tf.concat([self.conv2, self.tf_weights],3)
        self.all_act = tf.layers.conv2d(
            inputs=concat,
            filters=1,
            kernel_size=[1, 1],
            activation=tf.nn.relu,
            kernel_initializer = tf.contrib.layers.xavier_initializer(),
            name='conv3'
        )

        self.all_act_prob = tf.nn.softmax(self.all_act, dim=1, name='computed_weights')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.all_act_prob, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #loss = tf.reduce_sum(-tf.log(neg_log_prob)*self.tf_vt, axis=1)
            #loss = tf.reduce_sum(neg_log_prob * self.tf_vt)  # reward guided loss
            #self.last_weights = tf.gather_nd(self.tf_vt, [[[19]]])
#            last_weights = tf.gather_nd(last_weights, [49])

            self.w_t_1 = tf.gather_nd(self.tf_weights, [[[19]]])
            self.weights_diff = tf.subtract(self.w_t_1, self.all_act_prob, name='subbssss')
            mu = 1 #0.002*tf.reduce_sum(tf.abs(self.weights_diff))
            self.loss = tf.losses.compute_weighted_loss(mu*self.all_act_prob, -self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def choose_weights(self, prices, historic_weights):
        #normalize the input
        weights = np.array(historic_weights).reshape(1,6,1,20)
        prob_weights, conv2, conv1, transposed_prices = self.sess.run((self.all_act_prob, self.conv2, self.conv1, self.prices),
                                              feed_dict={self.tf_obs: prices, self.tf_weights: weights})
                                                                       #observation.values.flatten().tolist()})
                                                                       #observation[np.newaxis, :]})
        return prob_weights.flatten()

    def store_transition(self, s, l, f):
        self.ep_prices.append(s)
        self.ep_last_weights.append(l)
        self.ep_fp.append(f)

    def learn(self):
        if len(self.ep_prices) != 50: return

        future_prices = np.array(self.ep_fp).reshape((50,6,1,1))
        weights = np.array(self.ep_last_weights).reshape(50,6,1,20)
        old_and_current_prices = np.vstack(self.ep_prices)

# train on episode
        for i in range (0, 100):
            _ , cost, weights_diff = self.sess.run((self.train_op, self.loss, self.weights_diff), feed_dict={
                 self.tf_obs: old_and_current_prices,
                 self.tf_weights: weights,
                 self.tf_vt: future_prices
            })
            print("Cost[", i,"]:", cost)

        self.ep_prices, self.ep_last_weights, self.ep_fp = [], [], []    # empty episode data
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, "logs/model.ckpt")
        print("Model saved in file: %s" % save_path)



