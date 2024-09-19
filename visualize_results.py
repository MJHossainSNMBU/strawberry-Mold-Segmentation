
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope

H = 512
W = 512

# Define your custom metrics if any (assuming dice_coef and dice_loss are defined)
def dice_coef(y_true, y_pred):
    return tf.reduce_mean((2. * y_true * y_pred) / (y_true + y_pred + tf.keras.backend.epsilon()))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

model_path = '/content/data_folder/model.h5'

def visualize_predictions(model, test_x, test_y, num_samples=3):
    indices = random.sample(range(len(test_x)), num_samples)
    for i in indices:
        x, y = test_x[i], test_y[i]

        # Extracting the name
        name = x.split("/")[-1]

        # Reading the image
        image = cv2.imread(x, cv2.IMREAD_COLOR)   # [H, W, 3] in BGR format
        image = cv2.resize(image, (W, H))         # Resize to target dimensions
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        x = image_rgb / 255.0                     # Normalize image
        x = np.expand_dims(x, axis=0)             # Add batch dimension [1, H, W, 3]

        # Reading the mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))

        # Prediction
        y_pred = model.predict(x, verbose=0)[0]   # Predict mask
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5                    # Threshold prediction
        y_pred = y_pred.astype(np.int32)

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image_rgb)                     # Display in RGB format
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(y_pred, cmap='gray')
        plt.axis('off')

        plt.show()

# Load the best model
with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
    model = tf.keras.models.load_model(model_path)

# Visualize three random sample predictions
visualize_predictions(model, test_x, test_y, num_samples=3)
