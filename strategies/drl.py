import re
from .base import Base
import core.common as common
from .enums import TradeState
from core.bots.enums import BuySellMode
from core.tradeaction import TradeAction

import numpy as np
import pandas as pd
from strategies.DRL_brain import PolicyGradient


class Drl(Base):
    """
    About: Multi-currency strategy focusing on using deep reinforcement learning algorithms
    """
    last_weights = []
    last_prices = None
    initial_portfolio_value = None
    rewards = []
    step = 0

    def __init__(self):
        super(Drl, self).__init__()
        self.name = 'drl'
        self.min_history_ticks = 50
        self.buy_sell_mode = BuySellMode.user_defined
        self.DRL = PolicyGradient(
            n_actions=6,
            n_features=6*self.min_history_ticks,
            learning_rate=0.02,
            reward_decay=0.99,
            output_graph=True,
        )

    def compute_relevant_pairs(self, wallet, main_currency):
        currencies = list(wallet.current_balance.keys())
        for i in range(0, len(currencies)):
            currencies[i] = main_currency + '_' + currencies[i]
        return currencies

    def compute_weights_and_value(self, look_back, wallet):
        """
        Compute the weigths of each currency in the wallet
        :param df: latest currency_pairs price, assumes the data frame includes the needed pairs
        :param wallet: the current wallet, assumes all amounts are expressed in each currency
        :return: the weights as a vector
        """

        currencies = list(wallet.current_balance.keys())
        amounts = list(wallet.current_balance.values())
        sum = 0;

        for i in range(0, len(currencies)):
            if(currencies[i] == 'BTC'):
                sum += amounts[i]
                continue
            amounts[i] = amounts[i] * self.get_price(TradeState.none, look_back, 'BTC_'+currencies[i])
            sum += amounts[i]

        # let's hide the negative weights from the lib
        weights = np.asarray(amounts) / sum
        weights.clip(0,1)

        return (weights, sum)

    '''
    def build_net(self, features):
        input_layer = tf.reshape(features["x"], [-1, 3, 11, 50])
        weights_stack = tf.placeholder(tf.float32, shape=(8, 11, 1 ))
        conv1_layer = tf.layers.conv2d(
            inputs=input_layer,
            filters=2,
            kernel_size=[1, 3],
            padding="same",
            activation=tf.nn.relu)
        conv2_layer = tf.layers.conv2d(
            inputs=conv1_layer,
            filters=1,
            kernel_size=[1, 48],
            padding="same",
            activation=tf.nn.relu)
        conv3_layer = tf.layers.conv2d(
            inputs=[conv2_layer],
            filters=1,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu)
        dense_layer = tf.layers.dense(
            inputs=conv2_layer,
            units=10,
            activation=tf.nn.tanh,  # tanh activation
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1')
    '''
    def compute_rewards(self, portfolio_value):
        return

    def calculate(self, look_back, wallet):
        """
        Main Strategy function, which takes recent history data and returns recommended list of actions
        """

        (dataset_cnt, pairs_count) = common.get_dataset_count(look_back, self.group_by_field)

        # Wait until we have enough data
        if dataset_cnt < self.min_history_ticks:
            print('dataset_cnt:', dataset_cnt)
            return self.actions

        self.actions.clear()

        look_back = look_back.tail(pairs_count * self.min_history_ticks)
        pairs_names = look_back.pair.unique()

        (current_weights, wallet_value) = self.compute_weights_and_value(look_back, wallet)
        current_amounts = list(wallet.current_balance.values())

        if not self.initial_portfolio_value:
            self.initial_portfolio_value = wallet_value
            self.step = 0

        if self.last_weights!=[]:
            self.step +=1
            reward = np.log(wallet_value/self.initial_portfolio_value)/self.step
            self.DRL.store_transition(self.last_prices,self.last_weights,reward)

        if self.step ==100:
            discounted_ep_rs_norm = self.DRL.learn()
            print("discounted_ep_rs_norm is: ", discounted_ep_rs_norm)
            self.initial_portfolio_value = None

        currencies = list(wallet.current_balance.keys())
        look_back.reset_index()
#        converted_look_back = look_back.pivot(index=['date', 'pair'], columns='pair', values='close')
        converted_look_back = look_back.pivot_table(index='date', columns='pair', values='close')

        converted_look_back['BTC_BTC'] = 1
        converted_look_back = converted_look_back.filter(items = self.compute_relevant_pairs(wallet, 'BTC'))
        converted_look_back = converted_look_back.tail(self.min_history_ticks)

        next_weights = self.DRL.choose_weights(converted_look_back)

        self.last_weights = next_weights
        self.last_prices = converted_look_back
#        next_weights = self.algo.next_weights(converted_look_back, current_weights)



        # We sell first, then we buy
        # if new_weight < current_weight, we need to sell the related currency
        for i in range(0, len(currencies)):
            if(next_weights[i] < current_weights[i]
                and currencies[i] != 'BTC'
                and current_amounts[i] > 0): # no need to sell BTC
                close_pair_price = self.get_price(TradeState.sell, look_back, 'BTC_'+currencies[i])
                action = TradeAction('BTC_'+currencies[i],
                                     TradeState.sell,
                                     amount=current_amounts[i] * (current_weights[i]-next_weights[i])/current_weights[i],
                                     rate=close_pair_price,
                                     buy_sell_mode=self.buy_sell_mode)
                self.actions.append(action)

        # if new_weight > current_weight, we need to buy the related currency
        for i in range(0, len(currencies)):
            if (next_weights[i] > current_weights[i]
                and currencies[i] != 'BTC'
                and next_weights[i] != 0):
                close_pair_price = self.get_price(TradeState.sell, look_back, 'BTC_' + currencies[i])
                action = TradeAction('BTC_'+currencies[i],
                                     TradeState.buy,
                                     # let's only buy 99% of what the algo recommands to avoid going negative with the fees
                                     amount= 0.99*(wallet_value/close_pair_price) * (next_weights[i]-current_weights[i]),
                                     rate=close_pair_price,
                                     buy_sell_mode=self.buy_sell_mode)
                self.actions.append(action)

        return self.actions

