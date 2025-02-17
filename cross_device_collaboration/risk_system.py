import pandas as pd
from sklearn.ensemble import IsolationForest

# Load user trading data (this could include time, volume, price, etc.)
data = pd.read_csv('user_trading_data.csv')  # Example columns: time, volume, price, etc.

# Feature Engineering: Use basic statistics as features
data['price_diff'] = data['price'].diff()
data['volume_change'] = data['volume'].diff()

# Prepare the features for anomaly detection
X = data[['price_diff', 'volume_change']].dropna()

# Fit the model (Isolation Forest)
model = IsolationForest(contamination=0.05)  # 5% anomaly rate
model.fit(X)

# Predict anomalies (1 = normal, -1 = anomaly)
anomalies = model.predict(X)

# Alert if anomaly is detected
if -1 in anomalies:
    print("Potential scam risk detected!")
else:
    print("No scam risks detected.")
