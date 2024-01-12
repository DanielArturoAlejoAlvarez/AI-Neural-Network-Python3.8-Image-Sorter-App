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
