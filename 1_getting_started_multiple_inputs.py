import tensorflow as tf
from tensorboard_logging import *

# data
input_features_data = [
[1,100],
[2,200],
[2,250],
[3,400]
]# input data
correct_answers_data = [20,30,40,50]# train data
nb_epoches = 1000

# model
trained_var1 = tf.Variable([0.1], tf.float32,name="trained_var1")
trained_var2 = tf.Variable([0.1], tf.float32,name="trained_var2")
trained_var3 = tf.Variable([0.1], tf.float32,name="trained_var3")
input_features = tf.placeholder(tf.float32,shape=(2),name="input_features")
linear_model = trained_var1 * input_features[0] + trained_var2 * input_features[1] + trained_var3

# error loss
correct_answers = tf.placeholder(tf.float32,name="correct_answers")
squared_deltas = tf.square(linear_model - correct_answers)
loss = tf.reduce_sum(squared_deltas)

# train
optimizer = tf.train.GradientDescentOptimizer(1e-8)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
logger = Logger('tmp/basic')

with tf.Session() as sess:
	sess.run(init)
	test_input_data = {input_features:input_features_data[0], correct_answers:correct_answers_data[0]}
	print "=== input_features:\t%s" % sess.run(input_features[1],test_input_data)
	print "=== linear_model:\t%s" % sess.run(linear_model, test_input_data)
	print "=== correct_answers:\t%s" % sess.run(correct_answers, test_input_data)
	print "=== linear_model - correct_answers:\t%s" % sess.run(linear_model - correct_answers, test_input_data)
	# exit()
	print "\n----------------- BEFORE TRAIN"
	print "trained_var1\t\t%s" % sess.run(trained_var1)
	print "trained_var2\t\t%s" % sess.run(trained_var2)
	print "trained_var3\t\t%s" % sess.run(trained_var3)
	print "correct answers:\t%s" % correct_answers_data[0]
	print "model answers:\t\t%s" % sess.run(linear_model, test_input_data)
	print "error loss\t%s" % sess.run(loss, test_input_data)

	print "\n----------------- TRAINING"
	for i in range(nb_epoches):
		for j in range(len(input_features_data)):
			input_feature_data = input_features_data[j]
			correct_answer_data = correct_answers_data[j]
  			sess.run(train, {input_features:input_feature_data, correct_answers:correct_answer_data})
		if i % (nb_epoches/10.0) == 0 or i < 10:
			print "error loss\t%10s\tstep: %d" % (sess.run(loss, test_input_data), i)
		if i % (nb_epoches/100.0) == 0:
			logger.log_scalar("loss",sess.run(loss, test_input_data),i)

  	print "\n----------------- AFTER TRAIN"
	curr_trained_var1, curr_trained_var2,curr_trained_var3, curr_loss  = sess.run([trained_var1, trained_var2,trained_var3, loss], test_input_data)
	print "trained_var1\t\t%s" % curr_trained_var1
	print "trained_var2\t\t%s" % curr_trained_var2
	print "trained_var3\t\t%s" % curr_trained_var3
	print "input_features_data:\t%s" % input_features_data
	print "correct answers:\t%s" % correct_answers_data
	print "model answers:\t\t%s" % sess.run(linear_model, test_input_data)
	print "error loss\t%s" % curr_loss

	test_input_features = [3,600]
	print "\n--- SAMPLE MODEL ANSWER ---"
	print "test_input_features\t\t%s" % test_input_features
	print "model answers:\t\t%s" % sess.run(linear_model, {input_features:test_input_features})
	
	writer = tf.summary.FileWriter("tmp/basic", sess.graph)