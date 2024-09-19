
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from preprocessing import tf_dataset
from unet_model import build_unet
from metrics import dice_loss, dice_coef

# Variables
batch_size = 4
lr = 1e-4
num_epochs = 100

# Data paths
train_image_dir = '/content/train2/images'
train_mask_dir = '/content/train2/masks'
valid_image_dir = '/content/val2/images'
valid_mask_dir = '/content/val2/masks'
test_image_dir = '/content/test2/images'
test_mask_dir = '/content/test2/masks'

# Dataset loading
(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(train_image_dir, train_mask_dir, valid_image_dir, valid_mask_dir, test_image_dir, test_mask_dir)

print(f"Train: 	{len(train_x)} - {len(train_y)}")
print(f"Valid: 	{len(valid_x)} - {len(valid_y)}")
print(f"Test: 	{len(test_x)} - {len(test_y)}")

# Dataset preparation
train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

# Local paths
model_path_local = os.path.join("/content/data_folder", "model.h5")
csv_path_local = os.path.join("/content/data_folder", "log.csv")

# Google Drive paths
drive_folder = '/content/drive/My Drive/Fruit/try15082024/unetfruit'
if not os.path.exists(drive_folder):
    os.makedirs(drive_folder)

local_folder = '/content/data_folder'
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

model_path_drive = os.path.join(drive_folder, "model.h5")
csv_path_drive = os.path.join(drive_folder, "log.csv")

# Build and compile model
model = build_unet((IMG_H, IMG_W, 3))
model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef])

# Define callbacks
callbacks = [
    ModelCheckpoint(model_path_local, verbose=1, save_best_only=True),
    ModelCheckpoint(model_path_drive, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    CSVLogger(csv_path_local),
    CSVLogger(csv_path_drive),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
]

# Train the model
model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    callbacks=callbacks
)
