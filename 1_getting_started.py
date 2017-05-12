import tensorflow as tf

# data
features = [1,2,3,4]# input data
predictions = [0,-1,-2,-3]# train data

# model
trained_var1 = tf.Variable([0.3], tf.float32)
trained_var2 = tf.Variable([-0.3], tf.float32)
in_pl = tf.placeholder(tf.float32)
linear_model = trained_var1 * in_pl + trained_var2

# error loss
train_pl = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - train_pl)
loss = tf.reduce_sum(squared_deltas)

# train
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)

	print "\n----------------- BEFORE TRAIN"
	print "trained_var1\t%s" % sess.run(trained_var1)
	print "trained_var2\t%s" % sess.run(trained_var2)
	print "wanted data\t%s" % predictions
	print "result data\t%s" % sess.run(linear_model, {in_pl:features})
	print "error loss\t%s" % sess.run(loss, {in_pl:features, train_pl:predictions})

	print "\n----------------- TRAINING"
	for i in range(1000):
  		sess.run(train, {in_pl:features, train_pl:predictions})
  		if i%100 == 0 or i < 10:
  			print "error loss\t%s\tstep: %d" % (sess.run(loss, {in_pl:features, train_pl:predictions}), i)

  	print "\n----------------- AFTER TRAIN"
	curr_trained_var1, curr_trained_var2, curr_loss  = sess.run([trained_var1, trained_var2, loss], {in_pl:features, train_pl:predictions})
	print "trained_var1\t%s" % curr_trained_var1
	print "trained_var2\t%s" % curr_trained_var2
	print "wanted data\t%s" % predictions
	print "result data\t%s" % sess.run(linear_model, {in_pl:features})
	print "error loss\t%s" % curr_loss
	
	writer = tf.summary.FileWriter("tmp/basic", sess.graph)