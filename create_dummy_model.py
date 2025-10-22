import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# ------------------------------------------------------------
# Create a dummy CNN model with random weights
# ------------------------------------------------------------
def create_model(input_shape=(224, 224, 3), num_classes=6):
    model = models.Sequential([
        layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------------------------------------------------------------
# Generate dummy training data
# ------------------------------------------------------------
num_samples = 300
num_classes = 6

X_dummy = np.random.rand(num_samples, 224, 224, 3)
y_dummy = tf.keras.utils.to_categorical(np.random.randint(0, num_classes, size=(num_samples,)), num_classes)

# ------------------------------------------------------------
# Train briefly so weights are not uniform
# ------------------------------------------------------------
model = create_model()
model.fit(X_dummy, y_dummy, epochs=2, batch_size=8, verbose=1)

# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
print("✅ Model saved as models/model.h5")
