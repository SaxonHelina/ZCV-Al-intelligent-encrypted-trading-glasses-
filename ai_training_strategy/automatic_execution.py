import time


# Simulate real-time market data
def get_real_time_market_data():
    # Simulating market fluctuations
    return np.random.uniform(100, 200)


# Real-time strategy execution function
def real_time_trading():
    capital = 10000  # Starting capital
    position = 0  # No initial position
    market_threshold = 0.05  # Adjust strategy if price changes by 5%

    while True:
        current_price = get_real_time_market_data()

        # Adjust the strategy based on the market price change
        if position == 0 and current_price < 150:  # Example: Buy if price is below 150
            position = capital / current_price  # Buy stock
            capital = 0  # Spend all capital
            print(f"BUY at {current_price:.2f}")
        elif position > 0 and current_price > 160:  # Sell if price rises above 160
            capital = position * current_price  # Sell at current price
            position = 0  # Clear position
            print(f"SELL at {current_price:.2f}")

        # Print portfolio status
        print(
            f"Current Capital: {capital:.2f} | Current Position: {position * current_price if position > 0 else 0:.2f}")

        # Sleep for 1 second to simulate real-time execution
        time.sleep(1)


# Start real-time trading
real_time_trading()
