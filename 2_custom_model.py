import numpy as np
import tensorflow as tf
# features = [tf.contrib.layers.real_valued_column("features", dimension=1)]
# estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
# or custom model
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  trained_var1 = tf.get_variable("trained_var1", [1], dtype=tf.float64)
  trained_var2 = tf.get_variable("trained_var2", [1], dtype=tf.float64)
  features = features['features']
  predictions = trained_var1*features + trained_var2
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(predictions - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=predictions,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

# define our data set
features = np.array([1., 2., 3., 4.])
predictions = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"features": features}, predictions, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# evaluate our model
print(estimator.evaluate(input_fn=input_fn, steps=10))