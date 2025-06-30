#!/usr/bin/env python
# coding: utf-8



# In[ ]:

import os
os.makedirs("model_checkpoints", exist_ok=True)
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

#get_ipython().run_line_magic('pylab', 'inline')
# In[20]:


from pathlib import Path
data_dir = "reconn_image"
data_dir = Path(data_dir)
train_path = data_dir / "train"
test_path = data_dir / "test1"


# In[15]:


image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[24]:


import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import datetime
from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow as tf


# In[28]:


batch_size = 3
img_height = 200
img_width = 200

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
)

class_names = val_data.class_names
print(class_names)


# In[30]:


plt.figure(figsize=(10, 10))
for images, labels in val_data.take(1):
  for i in range(3):
    ax = plt.subplot(1, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[78]:


from tensorflow.keras import layers

num_classes = 2

model = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(128,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16,4, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
  metrics=['accuracy'],)

logdir="logs"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1, write_images=True)

checkpoint_path = "model_checkpoints/cp-{epoch:02d}.keras"
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=False,
    save_freq='epoch',
    verbose=1
)


model.fit(train_data, validation_data=val_data, epochs=6, callbacks=[checkpoint_callback])
model.save("final_modell.keras")
model.evaluate(train_data)
model.evaluate(val_data)


# In[79]:


model.summary()

