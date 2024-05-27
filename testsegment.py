import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set your paths
MASK_DIR = '/home/n9560751/home/venv31seg2/segment/mask/100masks'
IMAGE_DIR = '/home/n9560751/home/venv31seg2/segment/100images'
IMG_WIDTH = 2560
IMG_HEIGHT = 1792
IMG_CHANNELS = 3
NUM_CLASSES = 4
PATCH_SIZE = 512  # Patch size for training
PATCH_WIDTH = 512
PATCH_HEIGHT = 512
# Data augmentation parameters
data_gen_args = dict(
    rotation_range=45,
    horizontal_flip=True,
    fill_mode='nearest',
)

# Image data generator for images
image_datagen = ImageDataGenerator(**data_gen_args)

# Image data generator for masks
mask_datagen = ImageDataGenerator(**data_gen_args)

# Load images and masks
image_ids = next(os.walk(IMAGE_DIR))[2]

X_train = []
Y_train = []

print('Loading images and masks')
for n, image_file in tqdm(enumerate(image_ids), total=len(image_ids)):
    image_path = os.path.join(IMAGE_DIR, image_file)
    
    # Load image
    img = imread(image_path)[:,:,:IMG_CHANNELS]
    
    # Load masks for each class
    mask = np.zeros((img.shape[0], img.shape[1], NUM_CLASSES), dtype=bool)
    for class_idx, class_name in enumerate(['ground', 'tree', 'sky', 'animal']):
        mask_file_name = f'{os.path.splitext(image_file)[0]}_{class_name}_mask.png'
        mask_path = os.path.join(MASK_DIR, mask_file_name)
        
        # Load mask
        mask_ = imread(mask_path)
        binary_mask = (mask_ == 255)
        mask[:, :, class_idx] = binary_mask
    
    # Generate patches and corresponding masks
    for i in range(0, img.shape[0] - PATCH_SIZE + 1, PATCH_SIZE):
        for j in range(0, img.shape[1] - PATCH_SIZE + 1, PATCH_SIZE):
            patch_img = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patch_mask = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            
            # Original data
            X_train.append(patch_img)
            Y_train.append(patch_mask)
            
            # Apply data augmentation for patches
            seed = np.random.randint(9999)
            augmented_img = image_datagen.random_transform(patch_img, seed=seed)
            augmented_mask = mask_datagen.random_transform(patch_mask, seed=seed)
            X_train.append(augmented_img)
            Y_train.append(augmented_mask)

X_train = np.array(X_train)
Y_train = np.array(Y_train)

#BUILD THE MODEL

inputs = tf.keras.layers.Input((PATCH_HEIGHT, PATCH_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)
#Contraction Path

c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.2)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.2)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-6
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    focal_loss = -alpha * (1 - pt) ** gamma * tf.math.log(pt)
    return tf.reduce_mean(focal_loss)
    
model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

model.summary()

# Model checkpoints
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=r'C:/home/n9560751/home/venv31seg2/segment/model/segmentationv8.keras', verbose=1, save_best_only=True)

#model.save('segmentation.h5')
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=0.2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    checkpointer
]

#model = tf.keras.models.load_model('segmentation.h5')
# Train the model
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=20, epochs=30, callbacks=callbacks)