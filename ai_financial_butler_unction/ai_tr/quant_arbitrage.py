import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Machine Learning Anomaly Detection
# ---------------------------
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

# Simulate 2D feature data (e.g., from preprocessed log features)
np.random.seed(42)
n_normal = 200
n_outliers = 20
X_normal = 0.3 * np.random.randn(n_normal, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(n_outliers, 2))
X_ml = np.vstack((X_normal, X_outliers))
df_ml = pd.DataFrame(X_ml, columns=['feature1', 'feature2'])
contamination_rate = float(n_outliers) / (n_normal + n_outliers)

# Isolation Forest
iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
df_ml['anomaly_iso'] = iso_forest.fit_predict(X_ml)

# One-Class SVM
ocsvm = OneClassSVM(nu=contamination_rate, kernel="rbf", gamma=0.1)
df_ml['anomaly_ocsvm'] = ocsvm.fit_predict(X_ml)

# Local Outlier Factor (LOF)
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination_rate)
df_ml['anomaly_lof'] = lof.fit_predict(X_ml)

print("Isolation Forest detected anomalies:", np.sum(df_ml['anomaly_iso'] == -1))
print("One-Class SVM detected anomalies:", np.sum(df_ml['anomaly_ocsvm'] == -1))
print("Local Outlier Factor detected anomalies:", np.sum(df_ml['anomaly_lof'] == -1))

plt.figure(figsize=(14, 4))
plt.subplot(1, 3, 1)
plt.scatter(X_ml[:n_normal, 0], X_ml[:n_normal, 1], c='blue', label='Normal')
plt.scatter(X_ml[n_normal:, 0], X_ml[n_normal:, 1], c='red', label='Outlier')
plt.title("Ground Truth")
plt.legend()

plt.subplot(1, 3, 2)
normal_idx = df_ml['anomaly_iso'] == 1
anomaly_idx = df_ml['anomaly_iso'] == -1
plt.scatter(X_ml[normal_idx, 0], X_ml[normal_idx, 1], c='blue', label='Normal')
plt.scatter(X_ml[anomaly_idx, 0], X_ml[anomaly_idx, 1], c='red', label='Anomaly')
plt.title("Isolation Forest")
plt.legend()

plt.subplot(1, 3, 3)
normal_idx = df_ml['anomaly_ocsvm'] == 1
anomaly_idx = df_ml['anomaly_ocsvm'] == -1
plt.scatter(X_ml[normal_idx, 0], X_ml[normal_idx, 1], c='blue', label='Normal')
plt.scatter(X_ml[anomaly_idx, 0], X_ml[anomaly_idx, 1], c='red', label='Anomaly')
plt.title("One-Class SVM")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------
# Deep Learning Anomaly Detection using LSTM Autoencoder
# ---------------------------
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Input

# Simulate sequential log data (e.g., time series features extracted from logs)
# Here we create a sine wave and add anomalies
timesteps = 1000
t = np.linspace(0, 50, timesteps)
sequence = np.sin(t) + 0.1 * np.random.randn(timesteps)
# Inject anomalies
anomaly_indices = np.random.choice(np.arange(100, 900), size=20, replace=False)
sequence[anomaly_indices] += np.random.uniform(3, 5, size=20)

# Prepare the data for LSTM (using sliding window)
window_size = 30

def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i+window_size])
    return np.array(windows)

X_seq = create_windows(sequence, window_size)
X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))

# Split data into training and testing sets (train only on normal data)
train_size = int(0.6 * X_seq.shape[0])
X_train = X_seq[:train_size]
X_test = X_seq[train_size:]

# Build LSTM Autoencoder
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1), return_sequences=True),
    LSTM(32, activation='relu', return_sequences=False),
    RepeatVector(window_size),
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    TimeDistributed(Dense(1))
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# Train the model on normal sequences
history = model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

# Compute reconstruction error on test data
X_test_pred = model.predict(X_test)
reconstruction_error = np.mean(np.abs(X_test_pred - X_test), axis=(1,2))

# Set a threshold based on reconstruction error quantile
threshold = np.quantile(reconstruction_error, 0.95)
print("Reconstruction error threshold:", threshold)

# Detect anomalies in test set
anomalies = reconstruction_error > threshold
print("Number of anomalies detected by LSTM Autoencoder:", np.sum(anomalies))

# Visualization of reconstruction error
plt.figure(figsize=(10,4))
plt.plot(reconstruction_error, label='Reconstruction Error')
plt.hlines(threshold, xmin=0, xmax=len(reconstruction_error), colors='red', label='Threshold')
plt.title("LSTM Autoencoder Reconstruction Error")
plt.xlabel("Window Index")
plt.ylabel("Error")
plt.legend()
plt.show()

# Visualize a few examples
n_examples = 5
plt.figure(figsize=(12, 8))
for i in range(n_examples):
    idx = np.random.randint(0, X_test.shape[0])
    plt.subplot(n_examples, 1, i+1)
    plt.plot(X_test[idx].flatten(), label='Original')
    plt.plot(X_test_pred[idx].flatten(), label='Reconstruction')
    plt.title(f"Example {i+1} - {'Anomaly' if reconstruction_error[idx] > threshold else 'Normal'}")
    plt.legend()
plt.tight_layout()
plt.show()



class QuantArbitrage:
    """Executes quantitative arbitrage strategies based on user risk preferences."""

    def __init__(self, risk_tolerance):
        self.risk_tolerance = risk_tolerance

    def adjust_strategy(self):
        """Adjusts the arbitrage strategy based on market conditions."""
        # Placeholder for strategy adjustment logic
        print(f"Adjusting strategy for risk tolerance: {self.risk_tolerance}")

    def execute_strategy(self):
        """Executes the adjusted arbitrage strategy."""
        print("Executing arbitrage strategy.")

# Example usage
if __name__ == "__main__":
    arbitrage = QuantArbitrage(risk_tolerance='medium')
    arbitrage.adjust_strategy()
    arbitrage.execute_strategy()
