import tensorflow as tf

# data
input_features_data = [1,2,3,4]# input data
correct_answers_data = [0,-1,-2,-3]# train data

# model
trained_var1 = tf.Variable([0.3], tf.float32,name="trained_var1")
trained_var2 = tf.Variable([-0.3], tf.float32,name="trained_var2")
input_features = tf.placeholder(tf.float32,name="input_features")
linear_model = trained_var1 * input_features + trained_var2

# error loss
correct_answers = tf.placeholder(tf.float32,name="correct_answers")
squared_deltas = tf.square(linear_model - correct_answers,name="squared_deltas")
loss = tf.reduce_sum(squared_deltas,name="loss")

# train
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	print "\n----------------- BEFORE TRAIN"
	print "trained_var1\t\t%s" % sess.run(trained_var1)
	print "trained_var2\t\t%s" % sess.run(trained_var2)
	print "correct answers:\t%s" % correct_answers_data
	print "model answers:\t\t%s" % sess.run(linear_model, {input_features:input_features_data})
	print "error loss\t%s" % sess.run(loss, {input_features:input_features_data, correct_answers:correct_answers_data})

	print "\n----------------- TRAINING"
	for i in range(1000):
  		sess.run(train, {input_features:input_features_data, correct_answers:correct_answers_data})
  		if i%100 == 0 or i < 10:
  			print "error loss\t%10s\tstep: %d" % (sess.run(loss, {input_features:input_features_data, correct_answers:correct_answers_data}), i)

  	print "\n----------------- AFTER TRAIN"
	curr_trained_var1, curr_trained_var2, curr_loss  = sess.run([trained_var1, trained_var2, loss], {input_features:input_features_data, correct_answers:correct_answers_data})
	print "trained_var1\t\t%s" % curr_trained_var1
	print "trained_var2\t\t%s" % curr_trained_var2
	print "correct answers:\t%s" % correct_answers_data
	print "model answers:\t\t%s" % sess.run(linear_model, {input_features:input_features_data})
	print "error loss\t%s" % curr_loss

	test_input_features = [4,5,7,10]
	print "\n--- SAMPLE MODEL ANSWER ---"
	print "test_input_features\t\t%s" % test_input_features
	print "model answers:\t\t%s" % sess.run(linear_model, {input_features:test_input_features})
	
	writer = tf.summary.FileWriter("tmp/basic", sess.graph)