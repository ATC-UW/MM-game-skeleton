import numpy as np
import matplotlib.pyplot as plt

# Market Parameters - Adjusted for more realistic behavior
initial_price = 100.0
P_fundamental = 100.0
drift = 0.0005              # Reduced drift (0.05% daily)
volatility = 0.01           # Reduced volatility (1% daily)
theta = 0.05               # Reduced mean reversion speed
price_impact = 0.00001     # Significantly reduced price impact
volume_impact = 0.000001   # Reduced volume impact
s_base = 0.0005           # Tighter base spread (5 basis points)
k = 0.1                   # Reduced volatility impact on spread

# Simulation Parameters
N = 252                    # Trading days
np.random.seed()         # For reproducibility

def simulate_market():
    prices = np.zeros((N, 3))  # [bid, ask, mid]
    current_price = initial_price
    
    # Add some randomness to initial volumes
    volume_buy = np.random.poisson(5000)
    volume_sell = np.random.poisson(5000)
    
    for i in range(N):
        # Add some noise to fundamental price
        current_fundamental = P_fundamental * (1 + np.random.normal(0, 0.0001))
        
        # Generate random components with smoothing
        epsilon = np.random.normal(0, 1.0)
        if i > 0:
            epsilon = 0.7 * epsilon + 0.3 * np.random.normal(0, 1.0)
        
        # Calculate log return with smoother mean reversion
        log_return = (drift - 0.5 * volatility * volatility + 
                     volatility * epsilon + 
                     theta * (np.log(current_fundamental) - np.log(current_price)))
        
        # Update price with smoothing
        P_t = current_price * np.exp(log_return)
        
        # Generate more realistic order book volumes
        volume_buy = 0.8 * volume_buy + 0.2 * np.random.poisson(5000)
        volume_sell = 0.8 * volume_sell + 0.2 * np.random.poisson(5000)
        
        # Calculate adjusted price with dampened impact
        order_imbalance = (volume_buy - volume_sell) / (volume_buy + volume_sell)
        P_adjusted = P_t * (1 + price_impact * order_imbalance)
        
        # Calculate spread with dampening
        relative_spread = s_base + k * volatility * abs(epsilon) + volume_impact * abs(order_imbalance)
        spread = P_adjusted * relative_spread
        
        # Store prices
        prices[i, 0] = P_adjusted - spread/2  # bid
        prices[i, 1] = P_adjusted + spread/2  # ask
        prices[i, 2] = P_adjusted             # mid
        
        current_price = P_adjusted
    
    return prices

# Run simulation
prices = simulate_market()

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(prices[:, 0], label='Bid Price', color='blue', alpha=0.8)
plt.plot(prices[:, 1], label='Ask Price', color='red', alpha=0.8)
plt.plot(prices[:, 2], label='Mid Price', color='green', alpha=0.5)
plt.axhline(y=P_fundamental, color='black', linestyle='--', alpha=0.3, label='Fundamental Price')

plt.title('Market Prices Simulation')
plt.xlabel('Time (Trading Days)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Add spread visualization
plt.figure(figsize=(14, 4))
spreads = prices[:, 1] - prices[:, 0]
plt.plot(spreads, label='Bid-Ask Spread', color='purple')
plt.title('Bid-Ask Spread Over Time')
plt.xlabel('Time (Trading Days)')
plt.ylabel('Spread')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()