from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib

import tensorflow as tf
import numpy as np

IRIS_TRAINING = "tmp/iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "tmp/iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# If the training and test sets aren't stored locally, download them.
if not os.path.exists(IRIS_TRAINING):
	raw = urllib.urlopen(IRIS_TRAINING_URL).read()
	with open(IRIS_TRAINING, "w") as f:
		f.write(raw)

if not os.path.exists(IRIS_TEST):
	raw = urllib.urlopen(IRIS_TEST_URL).read()
	with open(IRIS_TEST, "w") as f:
		f.write(raw)

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TRAINING,
    target_dtype=np.int,
    features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=IRIS_TEST,
    target_dtype=np.int,
    features_dtype=np.float32)


# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, #The set of feature columns defined above.
                                            hidden_units=[10, 20, 10], # Three hidden layers, containing 10, 20, and 10 neurons, respectively.
                                            n_classes=3, #Three target classes, representing the three Iris species.
                                            model_dir="tmp/iris_model") #The directory in which TensorFlow will save checkpoint

# Define the test inputs
def get_train_inputs():
  x = tf.constant(training_set.data)
  y = tf.constant(training_set.target)

  return x, y

def printData():
	# Evaluate accuracy.
	accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
	                                     steps=1)["accuracy"]
	print("\nTest Accuracy: {0:f}\n".format(accuracy_score))
	predictions = list(classifier.predict_classes(input_fn=new_samples))
	print(
	    "New Samples, Class Predictions:    {}\n"
	    .format(predictions))

# Classify two new flower samples.
def new_samples():
  return np.array(
    [[6.4, 3.2, 4.5, 1.5],
     [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  # Define the test inputs
def get_test_inputs():
  x = tf.constant(test_set.data)
  y = tf.constant(test_set.target)

  return x, y


classifier.fit(input_fn=get_train_inputs, steps=1)

print("\n----------------- BEFORE TRAIN")
printData()

print("\n----------------- TRAINING")
tf.logging.set_verbosity(tf.logging.INFO)
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)
classifier.fit(input_fn=get_train_inputs,
	steps=600,
	monitors=[validation_monitor]
	)

print("\n----------------- AFTER TRAIN")
printData()


