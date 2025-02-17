from Crypto.Cipher import AES
import os
import base64
import hashlib
from sklearn.neighbors import LocalOutlierFactor


# AES Encryption and Decryption Class
class Encryption:
    def __init__(self, key):
        self.key = hashlib.sha256(key.encode()).digest()  # Create a 256-bit AES key from the given password

    def encrypt(self, data):
        cipher = AES.new(self.key, AES.MODE_CBC)  # Create cipher object
        padded_data = data + (16 - len(data) % 16) * ' '  # Padding the data
        encrypted_data = cipher.encrypt(padded_data.encode())
        return base64.b64encode(encrypted_data).decode()  # Return encrypted data as base64

    def decrypt(self, encrypted_data):
        cipher = AES.new(self.key, AES.MODE_CBC)
        encrypted_data_bytes = base64.b64decode(encrypted_data)
        decrypted_data = cipher.decrypt(encrypted_data_bytes)
        return decrypted_data.decode().strip()  # Return decrypted data after removing padding


# Risk assessment with AI - Detect unusual trading behavior
def assess_risk(transaction_data):
    # Example of transaction data with features (e.g., transaction amount, frequency, etc.)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    predictions = model.fit_predict(transaction_data)

    if -1 in predictions:  # -1 indicates an anomaly
        print("Risk detected: Anomalous behavior identified.")
    else:
        print("Behavior is normal.")


# Sample usage of encryption
encryption = Encryption(key="SuperSecureKey123")
transaction_data = "Sensitive Transaction Details"
encrypted_data = encryption.encrypt(transaction_data)
print(f"Encrypted Data: {encrypted_data}")

# Decrypt data when needed
decrypted_data = encryption.decrypt(encrypted_data)
print(f"Decrypted Data: {decrypted_data}")

# Sample transaction data for risk assessment
transaction_data = np.array([[1000, 5], [1500, 10], [2000, 2], [3000, 8]])  # Example features
assess_risk(transaction_data)
