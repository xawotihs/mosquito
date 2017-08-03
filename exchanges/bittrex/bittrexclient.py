import time
import pandas as pd
from bittrex.bittrex import Bittrex
from core.bots.enums import TradeMode
from exchanges.base import Base
from strategies.enums import TradeState
from termcolor import colored


class BittrexClient(Base):
    """
    Bittrex interface
    """

    def __init__(self, args, verbosity=2):
        super(BittrexClient, self).__init__()
        api_key = args['api_key']
        secret = args['secret']
        self.bittrex = Bittrex(api_key, secret)
        # self.buy_order_type = args['buy_order_type']
        # self.sell_order_type = args['sell_order_type']
        self.verbosity = verbosity

    def get_pairs(self):
        """
        Returns ticker pairs for all currencies
        """
        markets = self.bittrex.get_market_summaries()
        res = markets['result']
        pairs = []
        for market in res:
            pair = market['MarketName']
            pair = pair.replace('-', '_')
            pairs.append(pair)
        return pairs

    def return_candles(self, currency_pair, period=False, start=False, end=False):
        # TODO
        """
        Returns candlestick chart data
        """
        return self.bittrex.returnChartData(currency_pair, period, start, end)







    def get_balances(self):
        # TODO
        """
        Return available account balances (function returns ONLY currencies > 0)
        """
        try:
            balances = self.bittrex.returnBalances()
            only_non_zeros = {k: float(v) for k, v in balances.items() if float(v) > 0.0}
        except PoloniexError as e:
            print(colored('Got exception (polo.get_balances): ' + str(e), 'red'))
            only_non_zeros = dict()

        return only_non_zeros

    def get_symbol_ticker(self, symbol):
        # TODO
        """
        Returns real-time ticker Data-Frame
        """
        ticker = self.bittrex.returnTicker()[symbol]
        df = pd.DataFrame.from_dict(ticker, orient="index")
        df = df.T
        # We will use 'last' price as closing one
        df = df.rename(columns={'last': 'close', 'baseVolume': 'volume'})
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['pair'] = symbol
        df['date'] = int(time.time())
        return df

    def return_ticker(self):
        # TODO
        """
        Returns ticker for all currencies
        """
        return self.bittrex.returnTicker()

    def cancel_order(self, order_number):
        # TODO
        """
        Cancels order for given order number
        """
        return self.bittrex.cancelOrder(order_number)

    def return_open_orders(self, currency_pair='all'):
        # TODO
        """
        Returns your open orders
        """
        return self.bittrex.returnOpenOrders(currency_pair)

    def trade(self, actions, wallet, trade_mode):
        # TODO
        if trade_mode == TradeMode.backtest:
            return Base.trade(actions, wallet, trade_mode)
        else:
            actions = self.life_trade(actions)
            return actions

    def life_trade(self, actions):
        # TODO
        """
        Places orders and returns order number
        !!! For now we are NOT handling postOnly type of orders !!!
        """
        for action in actions:
            if self.verbosity > 0:
                print('Processing live-action: ' + str(action.action) +
                      ', amount:', str(action.amount) +
                      ', pair:', action.pair +
                      ', rate:', str(action.rate) +
                      ', buy_sell_all:', action.buy_sell_all)
            if action.action == TradeState.none:
                actions.remove(action)
                continue

            # Handle buy_sell_all cases
            wallet = self.get_balances()
            if action.buy_sell_all:
                action.amount = self.get_buy_sell_all_amount(wallet, action.action, action.pair, action.rate)

            # If we don't have enough assets, just skip/remove the action
            if action.amount == 0.0:
                print(colored('No assets to buy/sell, ...skipping: ' + str(action.amount) + action.pair, 'green'))
                actions.remove(action)
                continue

            # ** Buy Action **
            if action.action == TradeState.buy:
                try:
                    print(colored('setting buy order: ' + str(action.amount) + '' + action.pair, 'green'))
                    action.order_number = self.bittrex.buy(action.pair, action.rate, action.amount, self.buy_order_type)
                except PoloniexError as e:
                    print(colored('Got exception: ' + str(e) + 'txn: buy-' + action.pair, 'red'))
                    continue
                amount_unfilled = action.order_number.get('amountUnfilled')
                if amount_unfilled == 0.0:
                    actions.remove(action)
                else:
                    action.amount = amount_unfilled
            # ** Sell Action **
            elif action.action == TradeState.sell:
                try:
                    print(colored('setting sell order: ' + str(action.amount) + '' + action.pair, 'red'))
                    action.order_number = self.bittrex.sell(action.pair, action.rate,  action.amount, self.buy_order_type)
                except PoloniexError as e:
                    print(colored('Got exception: ' + str(e) + 'txn: sell-' + action.pair, 'red'))
                    continue
                amount_unfilled = action.order_number.get('amountUnfilled')
                if amount_unfilled == 0.0:
                    actions.remove(action)
                else:
                    action.amount = amount_unfilled
        return actions

    @staticmethod
    def get_buy_sell_all_amount(wallet, action, pair, rate):
        # TODO
        """
        Calculates total amount for ALL assets in wallet
        """
        if action == TradeState.none:
            return 0.0

        if rate == 0.0:
            print(colored('Got zero rate!. Can not calc. buy_sell_amount for pair: ' + pair, 'red'))
            return 0.0

        (symbol_1, symbol_2) = tuple(pair.split('_'))
        amount = 0.0
        if action == TradeState.buy and symbol_1 in wallet:
            assets = wallet.get(symbol_1)
            amount = assets/rate
        elif action == TradeState.sell and symbol_2 in wallet:
            assets = wallet.get(symbol_2)
            amount = assets

        return amount