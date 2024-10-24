# Image Sorter App With AI Neural Network And Python

## Description

Software of Development using AI, Neural Network, Tensorflow, Keras, Matplotlib and Python.


![alt text](https://thinkingneuron.com/wp-content/uploads/2020/10/How-to-use-ANN-for-classification-in-python.png)

## Apps

Google Colab

## Tools

Matplotlib, Tensorflow, Keras,Numpy, Adam,etc

## Usage

```html
$ git clone https://github.com/DanielArturoAlejoAlvarez/AI-Neural-Network-Python3.8-Image-Sorter-App.git[NAME APP]



$ virtualenv env

$ source env/bin/activate

$ pip install -r requirements.txt

$ python3 app.py

```

Follow the following steps and you're good to go! Important:

![alt text](https://deeplearning.neuromatch.io/_images/W1D1_Tutorial1_222_0.png](https://sethna.lassp.cornell.edu/Sloppy/SloppyFigs/training.gif)

##### Deep learning is the subset of machine learning methods based on artificial neural networks

## Coding

### Create Model (AI Neural Network)

```python
...
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
...
```

### Graphical Interface (Matplotlib)

```python
...
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

def draw_value_array(i, arr_predictions,label_r):
  arr_predictions,label_r = arr_predictions[i],label_r[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  graphic = plt.bar(range(10),arr_predictions,color="#777777")
  plt.ylim([0,1])
  label_prediction = np.argmax(arr_predictions)
  graphic[label_prediction].set_color("red")
  graphic[label_r].set_color("blue")

files = 5
columns = 5
num_images = files*columns

plt.figure(figsize=(2*2*columns, 2*files))

for i in range(num_images):
  plt.subplot(files, columns*2, i*2+1)
  draw_image(i,predictions,labels_test,images_test)
  plt.subplot(files, columns*2, i*2+2)
  draw_value_array(i,predictions,labels_test)

image = images_test[10]
image = np.array([image])
prediction = model.predict(image)
...
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/DanielArturoAlejoAlvarez/AI-Neural-Network-Python3.8-Image-Sorter-App. This project is intended to be a safe, welcoming space for collaboration, and contributors are expected to adhere to the [Contributor Covenant](http://contributor-covenant.org) code of conduct.

## License

The gem is available as open source under the terms of the [MIT License](http://opensource.org/licenses/MIT).

```

```
