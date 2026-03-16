import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("deepfake_model.h5")

# Ask user for image path
image_path = input("Enter image path: ")

# Read image
img = cv2.imread(image_path)
img = cv2.resize(img, (128, 128))
img = img / 255.0
img = np.reshape(img, (1, 128, 128, 3))

# Predict
prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("Prediction: REAL")
else:
    print("Prediction: FAKE")