from .base import Base
from core.bots.enums import TradeMode
import time
import configargparse


class Paper(Base):
    """
    Main class for Paper trading
    """
    arg_parser = configargparse.get_argument_parser()
    arg_parser.add('--use_real_wallet', help='Use/not use fictive wallet (only for paper simulation)',
                   action='store_true')
    mode = TradeMode.paper
    ticker_df = None

    def __init__(self, wallet):
        args = self.arg_parser.parse_known_args()[0]
        super(Paper, self).__init__(self.mode)
        self.use_real_wallet = args.use_real_wallet
        if not self.use_real_wallet:
            self.balance = wallet.copy()

    def get_next(self, interval_in_min):
        """
        Returns next state
        Interval: Interval in minutes
        """
        interval_in_sec = interval_in_min*60
        epoch_now = int(time.time())
        if self.last_tick_epoch > 0:
            next_ticker_time = (self.last_tick_epoch + interval_in_sec)
            delay_second = epoch_now - next_ticker_time
            if delay_second < 0:
                print('Going to sleep for: ', abs(delay_second), ' seconds.')
                time.sleep(abs(delay_second))

        if not self.ticker_df.empty:
            self.ticker_df.drop(self.ticker_df.index, inplace=True)

        epoch_now = int(time.time())
        epoch_start = epoch_now - interval_in_sec*5  # just to be sure get extra 5 datasets
        epoch_end = epoch_now
        for pair in self.pairs:
            # print('getting candles for period:',  str(epoch_start) + '---' + str(epoch_end) + '----'
            #      + str(interval_in_sec) + pair)
            new_df = self.exchange.get_candles_df(pair, epoch_start, epoch_end, interval_in_sec)
            # rint('new_df____:', new_df)
            if self.ticker_df.empty:
                self.ticker_df = new_df.copy()
            else:
                self.ticker_df = self.ticker_df.append(new_df, ignore_index=True)
                # Remove duplicates
                # self.ticker_df.drop_duplicates(subset=['date', 'pair'], inplace=True, keep='last')

        self.last_tick_epoch = epoch_now
        return self.ticker_df.copy()

    def get_balance(self):
        """
        Returns wallet balance
        """
        if self.use_real_wallet:
            return self.exchange.get_balances()
        else:
            return self.balance.copy()

    def trade(self, actions, wallet, trades, force_sell=True):
        """
        Simulate currency buy/sell (places fictive buy/sell orders)
        """
        return super(Paper, self).trade(actions, wallet, trades, force_sell=True)
