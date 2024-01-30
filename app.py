import tensorflow as tf 
import numpy as np 
import tensorflow_datasets as tfds 
import matplotlib.pyplot as plt 
import math


""" Fashion-MNIST is a dataset of Zalando's 
    article images — consisting of a training set 
    of 60,000 examples and a test set of 10,000 examples 
    for benchmarking machine learning algorithms.
"""

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

metadata

data_training,data_test = data['train'],data['test']

class_nanes = metadata.features['label'].names

class_nanes

def normalize(images,labels):
  images=tf.cast(images,tf.float32)
  images /= 255 #Here passes it by 0-255 to 0-1
  return images,labels

data_training = data_training.map(normalize)
data_test = data_test.map(normalize)

data_training = data_training.cache()
data_test = data_test.cache()

for image,label in data_training.take(1):
  break

image = image.numpy().reshape((28,28))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10,10))

for i,(image,label) in enumerate(data_training.take(25)):
  image = image.numpy().reshape((28,28))
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid()
  plt.imshow(image, cmap=plt.cm.binary)
  plt.xlabel(class_nanes[label])

plt.show()

model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28,1)),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.relu),
  tf.keras.layers.Dense(50, activation=tf.nn.softmax)
])

model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  metrics=['accuracy']
)

num_ex_training=metadata.splits['train'].num_examples
num_ex_testing=metadata.splits['test'].num_examples

print(num_ex_training)
print(num_ex_testing)

LOT_SIZE=32

data_training=data_training.repeat().shuffle(num_ex_training).batch(LOT_SIZE)
data_test=data_test.batch(LOT_SIZE)

import math

history = model.fit(data_training, epochs=5, steps_per_epoch=math.ceil(num_ex_training/LOT_SIZE))

plt.xlabel("# Epochs")
plt.ylabel("Magnitude of the loss")
plt.plot(history.history["loss"])

for images_test,labels_test in data_test.take(1):
  images_test = images_test.numpy()
  labels_test = labels_test.numpy()
  predictions = model.predict(images_test)

def draw_image(i,arr_predictions,label_r,image):
  arr_predictions,label_r,img = arr_predictions[i],label_r[i],image[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img[...,0], cmap=plt.cm.binary)
  label_prediction=np.argmax(arr_predictions)
  if label_prediction == label_r:
    color = 'blue' #Ok prediction
  else:
    color = 'red' #error prediction

  plt.xlabel("{} {:2.0f}% ({})".format(
      class_nanes[label_prediction],
      100*np.max(arr_predictions),
      class_nanes[label_r]),
      color=color
  )
