import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize data to range [0, 1]
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

# One-hot encode labels
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# Define a CNN model for CIFAR-10 classification
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential(name="CIFAR10_CNN")
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model


input_shape = x_train.shape[1:]
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Define callbacks for training
checkpoint_dir = "./cifar10_checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_valacc_{val_accuracy:.2f}.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
early_stopping_cb = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
reduce_lr_cb = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

# Train the model
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_split=0.1,
                    callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
                    verbose=1)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: {:.4f}, Test accuracy: {:.4f}".format(test_loss, test_accuracy))

# Save the final model
model.save("cifar10_cnn_model.h5")


# Plot training history
def plot_training_history(history):
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], "b-", label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], "r-", label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_training_history(history)


# ---------------------------------------------------------
# Image Recognition Code using the Trained Model
# ---------------------------------------------------------
def preprocess_image(image_path, target_size):
    """
    Load an image from the specified path, resize it to target_size, and preprocess it.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_path):
    """
    Preprocess the image and predict its class using the provided model.
    Returns the predicted class and confidence score.
    """
    target_size = (32, 32)  # CIFAR-10 image dimensions
    img_array = preprocess_image(image_path, target_size)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    return predicted_class, confidence


# CIFAR-10 class labels
cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# Example usage for image recognition
example_image_path = 'example_image.jpg'  # Replace with the actual image path
if os.path.exists(example_image_path):
    predicted_class, confidence = predict_image(model, example_image_path)
    print("Predicted class: {} (confidence: {:.2f})".format(cifar10_labels[predicted_class], confidence))

    # Display the image with prediction
    img = load_img(example_image_path, target_size=(32, 32))
    plt.imshow(img)
    plt.title("Predicted: {} (confidence: {:.2f})".format(cifar10_labels[predicted_class], confidence))
    plt.axis("off")
    plt.show()
else:
    print("Image file '{}' not found.".format(example_image_path))

# Sample data representing user trading patterns (features: amount, time of day, market conditions, etc.)
data = pd.read_csv('user_trading_data.csv')

# Feature engineering based on user trading behavior
data['trade_duration'] = pd.to_datetime(data['trade_end_time']) - pd.to_datetime(data['trade_start_time'])
data['trade_rate'] = data['amount'] / data['trade_duration'].dt.seconds  # Example: amount per second

# Labels: 1 = Successful trade, 0 = Unsuccessful trade
X = data[['trade_rate', 'market_condition', 'trade_duration']]
y = data['trade_success']

# Train a model for personalized AI avatar (user's trading strategy)
model = RandomForestClassifier(n_estimators=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Personalized AI avatar is trained to simulate user's trading strategy
print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")
