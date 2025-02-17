import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Sample data representing transactions (columns: amount, transaction type, user ID, time, etc.)
data = pd.read_csv('user_transactions.csv')  # Example: 'amount', 'user_id', 'transaction_time', 'transaction_type'

# Feature Engineering: Add custom features like transaction rate, previous amount, etc.
data['transaction_rate'] = data['amount'] / data['amount'].shift(1)  # Change rate of transaction amount
data['transaction_time_diff'] = pd.to_datetime(data['transaction_time']).diff().dt.seconds  # Time difference

# Prepare features for the model
X = data[['transaction_rate', 'transaction_time_diff']].dropna()

# A simple machine learning model to classify normal vs risky transactions
model = RandomForestClassifier(n_estimators=100)
model.fit(X, data['is_risky'])  # 'is_risky' is a label indicating if the transaction was risky or not

# Real-time prediction (this could be updated as new transactions arrive)
new_transaction = {'transaction_rate': [0.25], 'transaction_time_diff': [30]}
new_data = pd.DataFrame(new_transaction)

prediction = model.predict(new_data)
if prediction[0] == 1:
    print("Risky transaction detected, freezing the transaction.")
else:
    print("Transaction is safe.")
