class Position:
    """
    Represents the current holdings of multiple assets.

    Attributes:
        holdings (dict): A dictionary of {symbol: size}.
        entry_prices (dict): A dictionary of {symbol: average_entry_price}.
        pl (float): Current profit/loss in cash.
    """
    def __init__(self):
        self.holdings = {}
        self.entry_prices = {}
        self.pl = 0.0

    def update(self, symbol, size, price):
        """
        更新持仓状态。
        """
        if symbol not in self.holdings:
            self.holdings[symbol] = 0
            self.entry_prices[symbol] = 0
        prev_size = self.holdings[symbol]
        new_size = prev_size + size
        if new_size != 0:
            self.entry_prices[symbol] = (self.entry_prices[symbol] * prev_size + price * size) / new_size
        else:
            self.entry_prices[symbol] = 0
        self.holdings[symbol] = new_size

    def close(self, symbol, exit_price):
        """
        关闭持仓。
        """
        if symbol not in self.holdings or self.holdings[symbol] == 0:
            raise ValueError(f"No position to close for {symbol}.")
        size = self.holdings[symbol]
        self.pl += (exit_price - self.entry_prices[symbol]) * size
        self.holdings[symbol] = 0

    def snapshot(self):
        """
        返回当前持仓的快照。
        """
        return {
            "holdings": self.holdings.copy(),
            "entry_prices": self.entry_prices.copy(),
            "pl": self.pl,
        }

    def __repr__(self):
        return f"<Position(holdings={self.holdings}, pl={self.pl})>"

    def optimize(self, symbol_list, weights):
        """
        Optimize positions based on target weights.

        Args:
            symbol_list (list): List of symbols to rebalance.
            weights (dict): Target weights for each symbol.
        """
        for symbol in symbol_list:
            if symbol in self.holdings:
                # Calculate target size based on weight
                target_size = weights.get(symbol, 0) * self.pl
                self.holdings[symbol] = target_size

    @property
    def is_long(self):
        return {k: v for k, v in self.holdings.items() if v > 0}

    @property
    def is_short(self):
        return {k: v for k, v in self.holdings.items() if v < 0}

    def __repr__(self):
        return f"<Position(holdings={self.holdings}, pl={self.pl})>"