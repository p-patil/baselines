import os

import tensorflow as tf
import keras
from swintransformer import SwinTransformer


BATCH_SIZE = 32
# IMG_SIZE = (160, 160)
IMG_SIZE = (224, 224)
NUM_CLASSES = 2


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_datagen = keras.preprocessing.image.ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)
train_dataset = tf.data.Dataset.from_generator(lambda: train_generator, tuple([tf.float32 for _ in range(NUM_CLASSES)]))
# train_dataset = train_generator

validation_datagen = keras.preprocessing.image.ImageDataGenerator()
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=IMG_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)
validation_dataset = tf.data.Dataset.from_generator(lambda: validation_generator, tuple([tf.float32 for _ in range(NUM_CLASSES)]))
# validation_dataset = validation_generator


model = tf.keras.Sequential([
  tf.keras.layers.Lambda(lambda data: keras.applications.imagenet_utils.preprocess_input(tf.cast(data, tf.float32), mode="torch"), input_shape=[*IMG_SIZE, 3]),
  SwinTransformer('swin_tiny_224', include_top=False, pretrained=True),
  tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])



base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

history = model.fit_generator(list(train_dataset), epochs=10, validation_data=validation_dataset)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig("plot.png")
