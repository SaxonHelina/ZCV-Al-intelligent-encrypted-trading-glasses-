import requests
import json

class DataAggregator:
    """Aggregates global financial market data and supports real-time subscriptions."""

    def __init__(self):
        self.subscribers = []

    def fetch_market_data(self):
        """Fetches the latest market data from a financial API."""
        url = 'https://api.example.com/marketdata'
        response = requests.get(url)
        if response.status_code == 200:
            market_data = response.json()
            self.notify_subscribers(market_data)
        else:
            print("Failed to fetch market data.")

    def subscribe(self, callback):
        """Allows users to subscribe to real-time market data updates."""
        self.subscribers.append(callback)

    def notify_subscribers(self, data):
        """Notifies all subscribers with the latest market data."""
        for callback in self.subscribers:
            callback(data)

# Example usage
def print_market_data(data):
    print("Received market data:", data)

if __name__ == "__main__":
    aggregator = DataAggregator()
    aggregator.subscribe(print_market_data)
    aggregator.fetch_market_data()
