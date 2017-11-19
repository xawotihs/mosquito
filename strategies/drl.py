import re
import configargparse
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
    arg_parser = configargparse.get_argument_parser()
    last_computed_weights = []
    last_prices = None
    last_20_measured_weights = []
    rewards = []
    step = 0


    def __init__(self):
        args = self.arg_parser.parse_known_args()[0]
        super(Drl, self).__init__()
        self.name = 'drl'
        self.min_history_ticks = 50
        self.buy_sell_mode = BuySellMode.user_defined
        self.DRL = None
        self.commission = float(args.polo_txn_fee)/100

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

    def compute_future_prices(self, look_back):
        '''
        We're computing normalized future prices compared to previous prices
        :param look_back:
        :return: a vector of normalized future prices
        '''
        close = look_back.pivot_table(index='pair', columns='date', values='close')
        close.loc['BTC_BTC'] = 1
 #       last_column = close.iloc[:,-1]
 #       close = close.div(last_column, axis='index')

        return close.iloc[:,-1]/close.iloc[:,-2]

    def calculate(self, look_back, wallet):
        """
        Main Strategy function, which takes recent history data and returns recommended list of actions
        """

        (current_weights, wallet_value) = self.compute_weights_and_value(look_back, wallet)
        print("Current weights:", current_weights)

        (dataset_cnt, pairs_count) = common.get_dataset_count(look_back, self.group_by_field)
        look_back = look_back.tail(pairs_count * self.min_history_ticks)
        pairs_names = look_back.pair.unique()
        look_back.reset_index()

        if not self.DRL:
            self.DRL = PolicyGradient(
                n_actions=pairs_count,
                n_features=3,#6*self.min_history_ticks
                output_graph=True,
            )


        if self.last_computed_weights!=[]:
            self.step +=1
            future_prices = self.compute_future_prices(look_back)
            weights = np.array(self.last_20_measured_weights).transpose((1,2,0))
            self.DRL.store_transition(
                 self.last_prices,
                 weights,
                 #np.array(self.last_20_measured_weights).reshape((pairs_count,1,20)),
                 future_prices,
                 np.array(current_weights).reshape((pairs_count+1,1,1)))

        if(len(self.last_20_measured_weights)>=20):
            self.last_20_measured_weights.pop()

        self.last_20_measured_weights.insert(0, current_weights[1:].reshape((pairs_count,1)))

        # Wait until we have enough data
        if dataset_cnt < self.min_history_ticks:
            print('dataset_cnt:', dataset_cnt)
            return self.actions


        self.actions.clear()

        current_amounts = list(wallet.current_balance.values())

        self.DRL.learn()

        currencies = list(wallet.current_balance.keys())
#        converted_look_back = look_back.pivot(index=['date', 'pair'], columns='pair', values='close')
#        converted_look_back = look_back.pivot_table(index='date', columns='pair', values='close')
        close = look_back.pivot_table(index='pair', columns='date', values='close')
        last_column = close.iloc[:,-1]
        close = close.div(last_column, axis='index')

        high = look_back.pivot_table(index='pair', columns='date', values='high')
        high = high.div(last_column, axis='index')

        low = look_back.pivot_table(index='pair', columns='date', values='low')
        low = low.div(last_column, axis='index')

        features = np.array([close.as_matrix(), high.as_matrix(), low.as_matrix()])
        features = features[np.newaxis, :]
#        assert(features.shape == (1,3,pairs_count,50))

#        converted_look_back = converted_look_back.filter(items = self.compute_relevant_pairs(wallet, 'BTC'))
#        converted_look_back = converted_look_back.tail(self.min_history_ticks)

#        next_weights = self.DRL.choose_weights(converted_look_back)
        next_weights = self.DRL.choose_weights(features, self.last_20_measured_weights)
        print("New weights:", next_weights)
#        assert( (next_weights[0] != next_weights[1]) and (next_weights[1] != next_weights[2]) )

        self.last_computed_weights = next_weights[np.newaxis, :]
        self.last_prices = features

        # let's estimate mu
        mu, _ = self.DRL.compute_mu(next_weights, current_weights, self.commission)

        # We sell first, then we buy
        # if new_weight < current_weight, we need to sell the related currency
        for i in range(0, len(currencies)):
            if(next_weights[i] < current_weights[i]
                and currencies[i] != 'BTC'
                and current_amounts[i] > 0): # no need to sell BTC
                close_pair_price = self.get_price(TradeState.sell, look_back, 'BTC_'+currencies[i])
                action = TradeAction('BTC_'+currencies[i],
                                     TradeState.sell,
                                     amount=current_amounts[i] * (current_weights[i]-next_weights[i])/(current_weights[i]*mu),
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
                                     amount= mu*(wallet_value/close_pair_price) * (next_weights[i]-current_weights[i]),
                                     rate=close_pair_price,
                                     buy_sell_mode=self.buy_sell_mode)
                self.actions.append(action)

        return self.actions
