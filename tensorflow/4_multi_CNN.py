import tensorflow as tf
from helpers import *

from tensorflow.examples.tutorials.mnist import input_data


# utils
def printData():
    print "Weighs fully connected 2"
    tmp_var = sess.run(W_fc2)
    for i in range(10):
      print "%s \t\t\t\t id: %s" % (tmp_var[i*784/10],i*784/10)
    print "b_fc2 \n %s" % sess.run(b_fc2)
    testModelAndPrint()
    printErrorLos()


def testModelAndPrint():
    batch_xs, batch_ys = mnist.train.next_batch(10)
    print "correct answers:\t%s" % sess.run(tf.argmax(batch_ys, 1))
    print "model answers:\t\t%s" % sess.run(tf.argmax(model_conv,1) ,feed_dict={in_image_pix: batch_xs, keep_prob: 1.0})

def printErrorLos():
    print("test accuracy %g"%accuracy.eval(feed_dict={
    in_image_pix: mnist.test.images, correct_answers: mnist.test.labels, keep_prob: 1.0}))


print "\n----------------- Import data"
mnist = input_data.read_data_sets('tmp/mnist_input_data', one_hot=True)

print "\n----------------- Create the model"
in_image_pix = tf.placeholder(tf.float32, [None, 784])

#First Convolutional Layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(in_image_pix, [-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Second Convolutional Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

model_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


print "\n----------------- Define loss and optimizer"
correct_answers = tf.placeholder(tf.float32, [None, 10])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_answers, logits=model_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(model_conv,1), tf.argmax(correct_answers,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

print "\n----------------- BEFORE TRAIN"
printData()

print "\n----------------- TRAINING"
for i in range(100):
	batch = mnist.train.next_batch(20)
	train_accuracy = accuracy.eval(feed_dict={
	in_image_pix:batch[0], correct_answers: batch[1], keep_prob: 1.0})
	print("step %d, training accuracy %g"%(i, train_accuracy))
	train_step.run(feed_dict={in_image_pix: batch[0], correct_answers: batch[1], keep_prob: 0.5})



print "\n----------------- AFTER TRAIN"
printData()