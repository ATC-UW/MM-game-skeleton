from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Literal
import numpy as np

@dataclass
class OrderType:
    """
    Data class to represent the order type, which can be either "limit" or "market"
    """
    type: Literal["limit", "market"]
    from_time: int
    to_time: int

    def __init__ (self, type: Literal["limit", "market"], from_time: int, to_time: int):
        self.type = type
        self.from_time = from_time
        self.to_time = to_time

    def __str__(self):
        return f"OrderType(type={self.type}, from_time={self.from_time}, to_time={self.to_time})"
    
    def new_limit_order(from_time: int, to_time: int):
        """
        Create a new limit order with the given from_time and to_time
        """
        return OrderType("limit", from_time, to_time)
    
    def new_market_order(timestamp: int):
        """
        Create a new market order with the given timestamp
        """
        return OrderType("market", timestamp, timestamp)

class MarketMaker(ABC):
    """
    Abstract class for market maker, where each player will use previous interval data to make decisions
    to buy or sell stocks. The market maker will return the bid and ask prices, and the volume to buy and sell.
    The bid price represents the highest amount a buyer is willing to pay for a stock, 
    while the ask price describes the lowest level at which a seller is willing to sell their shares
    """
    @abstractmethod
    def update(self, prev_bid_price, prev_ask_price, holding, money, timestamp) -> Tuple[float, int, float, int, OrderType]:
        """
        Update the market maker with the previous bid and ask prices, and the timestamp of the current interval,
        and return the new bid and ask prices, and the volume to buy and sell

        There are two type of order you can make each time interval: market order and limit order.

        Market order: an order to buy or sell a stock at the current market price. The buy and sell price
        is determined by the market (not you). Market order doesn't have a time frame. If you are setting the
        next bid price and next ask price for a market order, it will be defaulted to the market bid and ask price.

        Limit order: an order to buy or sell a stock at a specific price or better. 
        A buy limit order can only be executed at the limit price or lower (compare to market ask/sell price), 
        and a sell limit order can only be executed at the limit price or higher (compare to market bid/buy price). 
        Limit order have a time frame, from_time and to_time, which specifies the time interval in which the
        order is valid.

        Market order have higher priority than limit order. If you are setting both market order and limit order
        in a given time interval, the market order will be executed first.

        :param prev_bid_price: the previous bid price
        :param prev_ask_price: the previous ask price
        :param holding: the number of stocks you are holding from previous interval
        :param money: the amount of money you have from previous interval
        :param timestamp: the timestamp of the current (not previous) interval

        :return: a tuple containing the new bid price limit, the volume to buy, the new ask price limit, 
        the volume to sell, and the order type as OrderType object
        """
        pass

class SimpleMarketMaker(MarketMaker):
    def __init__(self):
        self.price_history = []
        self.window_size = 20
        self.num_simulations = 30
        self.max_position = 50
        
    def simulate_future_price(self, current_price):
        if len(self.price_history) > self.window_size:
            # Calculate log returns
            returns = np.diff(np.log(self.price_history[-self.window_size:]))
            drift = np.mean(returns)
            vol = np.std(returns)
            
            # Simulate multiple price paths
            simulated_prices = []
            for _ in range(self.num_simulations):
                # Shorter prediction horizon (3 steps) for more accurate local predictions
                random_walk = np.random.normal(drift, vol, 3)
                price_path = current_price * np.exp(np.cumsum(random_walk))
                simulated_prices.append(price_path[-1])
            
            return np.mean(simulated_prices), np.percentile(simulated_prices, [25, 75])
        return current_price, (current_price * 0.99, current_price * 1.01)

    def update(self, prev_bid_price, prev_ask_price, holding, money, timestamp):
        mid_price = (prev_bid_price + prev_ask_price) / 2
        self.price_history.append(mid_price)
        
        # MCMC prediction
        expected_price, (lower_bound, upper_bound) = self.simulate_future_price(mid_price)
        price_range = upper_bound - lower_bound
        
        # Dynamic spread calculation
        base_spread = 0.3
        volatility_spread = price_range / mid_price * 0.5
        position_spread = abs(holding) / self.max_position * 0.4
        total_spread = base_spread + volatility_spread + position_spread
        
        # Position-based price adjustment
        position_skew = -holding / self.max_position * 0.3
        prediction_skew = (expected_price - mid_price) * 0.2
        
        # Set prices
        new_bid = mid_price - total_spread/2 + position_skew + prediction_skew
        new_ask = mid_price + total_spread/2 + position_skew + prediction_skew
        
        # Dynamic sizing based on prediction confidence and position
        base_size = 15
        confidence_factor = price_range / mid_price
        
        # Adjust sizes based on position and prediction
        if holding > 0:  # Long position
            if mid_price > expected_price:  # Predicted down move
                ask_size = int(base_size * (1.5 + confidence_factor))
                bid_size = int(base_size * 0.5)
            else:  # Predicted up move
                ask_size = int(base_size)
                bid_size = int(base_size * 0.5)
        else:  # Short position or neutral
            if mid_price < expected_price:  # Predicted up move
                bid_size = int(base_size * (1.5 + confidence_factor))
                ask_size = int(base_size * 0.5)
            else:  # Predicted down move
                bid_size = int(base_size)
                ask_size = int(base_size * 0.5)
        
        # Position limits
        if abs(holding) > self.max_position * 0.7:
            if holding > 0:
                bid_size = 5
                ask_size = max(ask_size, 30)
            else:
                ask_size = 5
                bid_size = max(bid_size, 30)
        
        # Money management
        max_notional = money * 0.2  # Don't use more than 20% of capital per order
        bid_size = min(bid_size, int(max_notional / new_bid))
        ask_size = min(ask_size, int(max_notional / new_ask))
        
        return new_bid, bid_size, new_ask, ask_size, OrderType.new_limit_order(timestamp, timestamp + 100)
    
