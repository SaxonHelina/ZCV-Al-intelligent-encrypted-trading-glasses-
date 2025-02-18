import time
import json
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, utils
from tensorflow.keras.datasets import mnist

# Blockchain implementation

class Block:
    def __init__(self, index, timestamp, data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            'index': self.index,
            'timestamp': self.timestamp,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self, difficulty=2):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "Genesis Block", "0")
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    def proof_of_work(self, block):
        block.nonce = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * self.difficulty):
            block.nonce += 1
            computed_hash = block.compute_hash()
        return computed_hash

    def add_block(self, block, proof):
        if self.last_block.hash != block.previous_hash:
            return False
        if not proof.startswith('0' * self.difficulty):
            return False
        block.hash = proof
        self.chain.append(block)
        return True

# Initialize blockchain with specified difficulty
blockchain = Blockchain(difficulty=2)

# Keras callback to log training metrics into the blockchain

class BlockchainCallback(callbacks.Callback):
    def __init__(self, blockchain):
        super().__init__()
        self.blockchain = blockchain

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        data = {
            'epoch': epoch + 1,
            'loss': logs.get('loss'),
            'accuracy': logs.get('accuracy'),
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy')
        }
        new_block = Block(
            index=self.blockchain.last_block.index + 1,
            timestamp=time.time(),
            data=data,
            previous_hash=self.blockchain.last_block.hash
        )
        proof = self.blockchain.proof_of_work(new_block)
        added = self.blockchain.add_block(new_block, proof)
        if added:
            print(f"Block added for epoch {epoch + 1}: {new_block.hash}")
        else:
            print("Failed to add block")

# Load MNIST dataset

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
num_classes = 10
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# Define a simple CNN model for MNIST classification

def create_model(input_shape, num_classes):
    model = models.Sequential(name="MNIST_CNN")
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

input_shape = x_train.shape[1:]
model = create_model(input_shape, num_classes)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

# Train the model with the blockchain callback

blockchain_callback = BlockchainCallback(blockchain)
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,
    validation_split=0.1,
    callbacks=[blockchain_callback],
    verbose=1
)

# Evaluate the model on test data

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

# Print the blockchain ledger

print("\nBlockchain Ledger:")
for block in blockchain.chain:
    print(f"Index: {block.index}, Timestamp: {block.timestamp}, Data: {block.data}, Hash: {block.hash}, Nonce: {block.nonce}")

# Image Recognition Code using the Trained Model

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, target_size):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = image_array.astype('float32') / 255.
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_image(model, image_path):
    target_size = (28, 28)  # MNIST image dimensions
    image_array = preprocess_image(image_path, target_size)
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_class, confidence

mnist_labels = [str(i) for i in range(10)]

example_image_path = 'example_mnist_image.png'  # Replace with an actual image path
if __name__ == '__main__':
    import os
    if os.path.exists(example_image_path):
        pred_class, conf = predict_image(model, example_image_path)
        print(f"Predicted class: {mnist_labels[pred_class]} with confidence: {conf:.2f}")
        image = load_img(example_image_path, target_size=(28, 28))
        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {mnist_labels[pred_class]} (Confidence: {conf:.2f})")
        plt.axis("off")
        plt.show()
    else:
        print(f"Image file '{example_image_path}' not found.")
