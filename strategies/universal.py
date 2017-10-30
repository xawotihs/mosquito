import re
from .base import Base
import core.common as common
from .enums import TradeState
from core.bots.enums import BuySellMode
from core.tradeaction import TradeAction

from universal import tools
from universal import algos
from universal.algos import *
import numpy
import pandas as pd


class Universal(Base):
    """
    About: Multi-currency strategy focusing on using universal portfolio management algorithms
    """
    first_values = pd.DataFrame()

    def __init__(self):
        super(Universal, self).__init__()
        self.name = 'universal'
        self.min_history_ticks = 5
        self.buy_sell_mode = BuySellMode.user_defined
        self.active_pairs = []
        self.algo = algos.OLMAR()

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


    def my_next_weights(self, S, last_b, **kwargs):
        """ Calculate weights for next day. """
        # use history in step method?
        use_history = self.algo._use_history_step()

        history = S / self.first_values


        #history = self.algo._convert_prices(S, self.algo.PRICE_TYPE)
        x = history.iloc[-1]

        if use_history:
            b = self.algo.step(x, last_b, history, **kwargs)
        else:
            b = self.algo.step(x, last_b, **kwargs)
        return pd.Series(b, index=S.columns)


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
        self.sync_active_pairs(wallet.current_balance)


        (current_weights, wallet_value) = self.compute_weights_and_value(look_back, wallet)
        current_amounts = list(wallet.current_balance.values())

        currencies = list(wallet.current_balance.keys())
        look_back.reset_index()
#        converted_look_back = look_back.pivot(index=['date', 'pair'], columns='pair', values='close')
        converted_look_back = look_back.pivot_table(index='date', columns='pair', values='close')

        converted_look_back['BTC_BTC'] = 1
        converted_look_back = converted_look_back.filter(items = self.compute_relevant_pairs(wallet, 'BTC'))
        converted_look_back = converted_look_back.tail(self.min_history_ticks)

        if self.first_values.empty:
            self.first_values = converted_look_back.iloc[0]
            self.algo.init_step(converted_look_back)

        next_weights = self.my_next_weights(converted_look_back, current_weights)
#        next_weights = (current_weights + next_weights) / 2
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

    def sync_active_pairs(self, wallet):
        """
        Synchronizes active_pairs container with current_balance
        """
        for active_pair in list(self.active_pairs):
            if active_pair not in wallet:
                self.active_pairs.remove(active_pair)
