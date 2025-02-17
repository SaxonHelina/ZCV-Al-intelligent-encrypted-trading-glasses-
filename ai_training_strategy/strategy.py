import numpy as np
import matplotlib.pyplot as plt

# Simulate historical market data (example data)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
market_data = pd.DataFrame({
    'date': dates,
    'market_price': np.random.uniform(100, 200, len(dates)),
})

# User-defined strategy (example: buy when price drops more than 10%)
def trading_strategy(data, buy_threshold=0.1):
    capital = 10000  # Starting capital
    position = 0  # No initial position
    trades = []

    for i in range(1, len(data)):
        # Buy when the market drops by more than buy_threshold from the previous day
        if data['market_price'].iloc[i] < data['market_price'].iloc[i - 1] * (1 - buy_threshold) and capital >= data['market_price'].iloc[i]:
            position = capital / data['market_price'].iloc[i]  # Buy the stock
            capital = 0  # Spend all capital
            trades.append((data['date'].iloc[i], 'BUY', data['market_price'].iloc[i]))

        # Sell if there is a profit
        elif position > 0 and data['market_price'].iloc[i] > data['market_price'].iloc[i - 1]:
            capital = position * data['market_price'].iloc[i]  # Sell at current price
            position = 0  # Clear position
            trades.append((data['date'].iloc[i], 'SELL', data['market_price'].iloc[i]))

    # Final capital and performance evaluation
    final_value = capital if position == 0 else position * data['market_price'].iloc[-1]
    return final_value, trades

# Backtest the strategy
final_value, trades = trading_strategy(market_data)

# Print out the trades
for trade in trades:
    print(f"Trade: {trade[0]} | Action: {trade[1]} | Price: {trade[2]}")

print(f"Final Portfolio Value: {final_value:.2f}")

# Plot the market data and trades
plt.plot(market_data['date'], market_data['market_price'], label='Market Price')
for trade in trades:
    plt.scatter(trade[0], trade[2], c='red' if trade[1] == 'BUY' else 'green', label=trade[1])
plt.legend()
plt.show()
