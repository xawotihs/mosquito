from .base import Base


class Ema(Base):
    """
    ema strategy
    """

    def __init__(self, args):
        super(Ema, self).__init__(args)
        self.name = 'ema'


    def calulate(self, interval):
        """
        Returns next state
        """

        print('running strategy ema')
        return 'buy'