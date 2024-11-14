from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Literal

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
    def update(self, prev_bid_price, prev_ask_price, timestamp) -> Tuple[float, int, float, int, OrderType]:
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
        :param timestamp: the timestamp of the current (not previous) interval

        :return: a tuple containing the new bid price limit, the volume to buy, the new ask price limit, 
        the volume to sell, and the order type as OrderType object
        """
        pass

class SimpleMarketMaker(MarketMaker):
        """
        An example on how to implement a market maker.
        """
        def __init__(self):
            self.spread = 0.02  # 2% spread
            self.order_size = 100  # Fixed order size
        
        # TODO: Replace this example with your strategy
        def update(self, prev_bid_price, prev_ask_price, timestamp) -> Tuple[float, int, float, int, OrderType]:
            """
            Example on how to implement the update method for the market maker.

            This example will return the previous bid price and ask price, and the volume to buy and sell
            as 100, and a new limit order with the timestamp and timestamp + 100.

            To return a market order, you can use OrderType.new_market_order(timestamp)
            ```
            return prev_bid_price, 100, prev_ask_price, 100, OrderType.new_market_order(timestamp)
            ```
            For market order, the limit price will be defaulted to the market bid and ask price.

            Note that the volume to buy and sell can be any integer value, including 0. If the volume is negative,
            it will be set to 0. If the volume to sell is more than the holding, it will be set to the holding.

            :param prev_bid_price: the previous bid price
            :param prev_ask_price: the previous ask price
            :param timestamp: the timestamp of the current (not previous) interval

            :return: a tuple containing the new bid price limit, the volume to buy, the new ask price limit,
            the volume to sell, and the order type as OrderType object

            """
            fair_price = (prev_bid_price + prev_ask_price) / 2
            
            # Set new bid and ask prices around fair price
            new_bid_price = fair_price * (1 - self.spread/2)
            new_ask_price = fair_price * (1 + self.spread/2)
            
            # Return tuple of (bid_price, bid_size, ask_price, ask_size, order_type)
            return (new_bid_price, self.order_size, new_ask_price, self.order_size, OrderType.new_limit_order(timestamp, timestamp + 10))


class BetterMarketMaker(MarketMaker):
    def __init__(self):
        # Basic parameters
        self.base_spread = 0.01  # 1% base spread
        self.max_inventory = 1000  # Maximum inventory we want to hold
        self.target_inventory = 500  # Ideal inventory level
        self.order_size = 50  # Base order size
        self.current_inventory = 0  # Track our inventory
        
    def calculate_dynamic_spread(self, inventory_ratio):
        """Adjust spread based on inventory position"""
        # Widen spread when inventory deviates from target
        return self.base_spread * (1 + abs(inventory_ratio) * 0.5)
    
    def calculate_skewed_prices(self, fair_price, spread, inventory_ratio):
        """Skew prices based on inventory position"""
        # If we have too much inventory, make selling more attractive
        # If we have too little, make buying more attractive
        mid_price_skew = fair_price * (1 + inventory_ratio * 0.01)
        half_spread = spread / 2
        
        bid_price = mid_price_skew * (1 - half_spread)
        ask_price = mid_price_skew * (1 + half_spread)
        return bid_price, ask_price
    
    def calculate_dynamic_sizes(self, inventory_ratio):
        """Adjust order sizes based on inventory position"""
        if inventory_ratio > 0:  # Too much inventory
            bid_size = int(self.order_size * (1 - inventory_ratio))
            ask_size = int(self.order_size * (1 + inventory_ratio))
        else:  # Too little inventory
            bid_size = int(self.order_size * (1 - inventory_ratio))
            ask_size = int(self.order_size * (1 + inventory_ratio))
        return max(10, bid_size), max(10, ask_size)  # Ensure minimum size of 10
    
    def update(self, prev_bid_price, prev_ask_price, timestamp) -> Tuple[float, int, float, int, OrderType]:
        # Calculate fair price from previous prices
        fair_price = (prev_bid_price + prev_ask_price) / 2
        
        # Calculate inventory ratio (-1 to 1) relative to target
        inventory_ratio = (self.current_inventory - self.target_inventory) / self.max_inventory
        
        # Get dynamic spread based on inventory position
        spread = self.calculate_dynamic_spread(inventory_ratio)
        
        # Calculate prices with inventory skew
        bid_price, ask_price = self.calculate_skewed_prices(fair_price, spread, inventory_ratio)
        
        # Calculate dynamic order sizes
        bid_size, ask_size = self.calculate_dynamic_sizes(inventory_ratio)
        
        # Update inventory (simplified simulation)
        self.current_inventory += bid_size - ask_size
        
        # Create limit order valid for next 10 timestamps
        order_type = OrderType.new_limit_order(timestamp, timestamp + 10)
        
        return bid_price, bid_size, ask_price, ask_size, order_type