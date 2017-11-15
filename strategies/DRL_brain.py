import numpy as np
import tensorflow as tf
from pathlib import Path
from core.tradeaction import TradeAction

# reproducible
#np.random.seed(1)
#tf.set_random_seed(1)
tf.logging.set_verbosity(tf.logging.INFO)


c = 0.0025
delta = 0.0000001


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.00003,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate

        self.ep_prices, self.ep_last_weights, self.ep_fp, self.ep_cp  = [], [], [], []

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

        my_file = Path("logs")
        if my_file.exists():
            # Restore variables from disk.
            self.saver.restore(self.sess, "logs/model.ckpt")

        self.sess.run(tf.global_variables_initializer())

    def compute_mu(self, new_w, old_w, cur = c):
        mu0 = cur * np.sum(np.abs(np.subtract(old_w[1:], new_w[1:])))
        k = 0
        while True:
            sub = np.subtract(old_w[1:], mu0 * new_w[1:])
            mu1 = (1 / (1 - cur * new_w[0])) * (1 - (cur * old_w[0]) - (2 * cur - cur * cur) * np.sum(np.maximum(sub, 0, sub)))
            if (abs(mu0 - mu1) > delta):
                mu0 = mu1
                k = k + 1
            else:
                break
        return (mu1).astype(np.float32), k

    def compute_mu_array(self, new_w, old_w):
        mu_array = np.array([])
        for r in range(new_w.shape[0]):
            (mu, k) = self.compute_mu(new_w[r,], old_w[r,])
            mu_array = np.append(mu_array, mu)

        return mu_array

    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features, self.n_actions, 50], name="observations")
            #From NCHW to NHWC
            #x = tf.reshape(self.tf_obs, [1, 2, 3, 4])  # input
            #x = tf.reshape(self.tf_obs, [-1, self.n_features, self.n_actions, 50])  # input
            self.prices = tf.transpose(self.tf_obs, [0, 2, 3, 1])
            self.prices = tf.Print(self.prices, [self.prices], "self.prices:", summarize=100)

            self.tf_vt = tf.placeholder(tf.float32, [None, self.n_actions+1, 1, 1], name="future_prices")
            self.tf_weights = tf.placeholder(tf.float32, [None, self.n_actions, 1, 20], name="historic_weights")
            self.tf_cp = tf.placeholder(tf.float32, [None, self.n_actions+1, 1, 1], name="current_weights")

        # CONV2D: stride of 1
        W1 = tf.get_variable("W1", [1, 3, 3, 2], initializer=tf.contrib.layers.xavier_initializer(seed=0))
        Z1 = tf.nn.conv2d(self.prices, W1, strides=[1, 1, 1, 1], padding='VALID')
        # RELU
        self.conv1 = tf.nn.relu(Z1)

#         self.conv1 = tf.layers.conv2d(
#             inputs=self.prices,
#             filters=2,
#             kernel_size=[1, 3],
#             activation=tf.nn.relu,
# #            kernel_initializer=tf.contrib.layers.xavier_initializer(),
# #            bias_initializer=tf.contrib.layers.xavier_initializer(),
# #            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00000001),
#             name='conv1',
#         )
        self.conv1 = tf.Print(self.conv1, [self.conv1], "self.conv1:", summarize=100)
        W2 = tf.get_variable("W2", [1, 48, 2, 1],
                             initializer=tf.contrib.layers.xavier_initializer(seed=0))
        Z2 = tf.nn.conv2d(self.conv1, W2, strides=[1, 1, 1, 1], padding='VALID')
        # RELU
        self.conv2 = tf.nn.relu(Z2)

        #         self.conv2 = tf.layers.conv2d(
#             inputs=self.conv1,
#             filters=1,
#             kernel_size=[1, 48],
#             activation=tf.nn.relu,
# #            kernel_initializer = tf.contrib.layers.xavier_initializer(),
# #            bias_initializer=tf.contrib.layers.xavier_initializer(),
# #            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00000001),
#             name='conv2',
#         )
        self.conv2 = tf.Print(self.conv2, [self.conv2], "self.conv2:", summarize=100)
        concat = tf.concat([self.conv2, self.tf_weights],3)
        concat = tf.Print(concat, [concat], "concat:", summarize=100)

        W3 = tf.get_variable("W3", [1, 1, 21, 1],
                             initializer=tf.contrib.layers.xavier_initializer(seed=0))
        Z3 = tf.nn.conv2d(concat, W3, strides=[1, 1, 1, 1], padding='VALID')
        # RELU
        self.conv3 = Z3 #tf.nn.relu(Z3)


#         self.conv3 = tf.layers.conv2d(
#             inputs=concat,
#             filters=1,
#             kernel_size=[1, 1],
#             activation=tf.nn.relu,
# #            kernel_initializer = tf.contrib.layers.xavier_initializer(),
# #            bias_initializer=tf.contrib.layers.xavier_initializer(),
# #            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.00000001),
#             name='conv3'
#         )
        self.conv3 = tf.Print(self.conv3, [self.conv3], "self.conv3:", summarize=100)

        cash_bias = tf.get_variable("cash_bias", [1], initializer=tf.contrib.layers.xavier_initializer())
        padded = tf.pad(self.conv3, tf.constant([[0,0], [1,0], [0,0], [0,0]]), "CONSTANT")
        #self.all_act = tf.concat([self.conv3, cash_bias],1)

        padded = tf.Print(padded, [padded], "padded:", summarize=12000)
        self.all_act_prob = tf.nn.softmax(padded+cash_bias, dim=1, name='computed_weights')  # use softmax to convert to probability

        with tf.name_scope('loss'):
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            #neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.all_act_prob, labels=self.tf_acts)   # this is negative log of chosen action
            # or in this way:
            #loss = tf.reduce_sum(-tf.log(neg_log_prob)*self.tf_vt, axis=1)
            #loss = tf.reduce_sum(neg_log_prob * self.tf_vt)  # reward guided loss
            #self.last_weights = self.tf_weights[:,:,:,19]
            #tf.gather_nd(self.tf_weights, [[[19]]])
            #last_weights = tf.gather_nd(last_weights, [49])

            #self.w_t_1 = tf.gather_nd(self.tf_weights, [[[19]]])
            #self.weights_diff = tf.subtract(self.w_t_1, self.all_act_prob, name='subbssss')
            mu_array = tf.py_func(self.compute_mu_array, [self.all_act_prob, self.tf_cp], tf.float64)
            mu_array = tf.cast(mu_array, tf.float32)  # [1, 2], dtype=tf.int32
            losses = tf.multiply(self.all_act_prob, mu_array)
            #mu = 1 #0.002*tf.reduce_sum(tf.abs(self.weights_diff))
            self.loss = tf.losses.compute_weighted_loss(losses, -self.tf_vt)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def choose_weights(self, prices, historic_weights):
        #normalize the input
        weights = np.array(historic_weights).reshape(1,self.n_actions,1,20)
        prob_weights, conv2, conv1, transposed_prices = self.sess.run((self.all_act_prob, self.conv2, self.conv1, self.prices),
                                              feed_dict={self.tf_obs: prices, self.tf_weights: weights})
                                                                       #observation.values.flatten().tolist()})
                                                                       #observation[np.newaxis, :]})
        return prob_weights.flatten()

    def store_transition(self, s, l, f, c):
        self.ep_prices.append(s)
        self.ep_last_weights.append(l)
        self.ep_fp.append(f)
        self.ep_cp.append(c)

    def learn(self):
        m = len(self.ep_prices)
        if m == 0 or m%100 != 0: return

        future_prices = np.array(self.ep_fp).reshape((m,self.n_actions+1,1,1))
        weights = np.array(self.ep_last_weights).reshape(m,self.n_actions,1,20)
        old_and_current_prices = np.vstack(self.ep_prices)
        current_prices = np.array(self.ep_cp).reshape((m,self.n_actions+1,1,1))

# train on episode
        randoms = np.random.randint(0, m-50, 100)
        for i in range (0, 100):
            _ , cost = self.sess.run((self.train_op, self.loss), feed_dict={
                self.tf_obs: old_and_current_prices[randoms[i]:randoms[i]+50, :, :, :],
                self.tf_weights: weights[randoms[i]:randoms[i]+50, :, :, :],
                self.tf_vt: future_prices[randoms[i]:randoms[i]+50, :],
                self.tf_cp: current_prices[randoms[i]:randoms[i]+50, :]
            })
            print("Cost[", i,"]:", cost)

#        self.ep_prices, self.ep_last_weights, self.ep_fp, self.ep_cp = [], [], [], []    # empty episode data
        # Save the variables to disk.
        save_path = self.saver.save(self.sess, "logs/model.ckpt")
        print("Model saved in file: %s" % save_path)


