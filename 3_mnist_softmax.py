from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy
numpy.set_printoptions(threshold=numpy.nan)


# utils
def printData():
    print "Weighs"
    tmp_var = sess.run(Weigh)
    for i in range(10):
      print "%s \t\t\t\t id: %s" % (tmp_var[i*784/10],i*784/10)
    print "bias \n %s" % sess.run(bias)
    testModelAndPrint()
    printErrorLos()


def testModelAndPrint():
    batch_xs, batch_ys = mnist.train.next_batch(10)
    print "correct answers:\t%s" % sess.run(tf.argmax(batch_ys, 1))
    print "model answers:\t\t%s" % sess.run(tf.argmax(model, 1), {in_image_pix: batch_xs})

def printErrorLos():
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(correct_answers, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print  "accuracy: %s" % (sess.run(accuracy, feed_dict={in_image_pix: mnist.test.images, correct_answers: mnist.test.labels}))



print "\n----------------- Import data"
mnist = input_data.read_data_sets('tmp/mnist_input_data', one_hot=True)

print "\n----------------- Create the model"
in_image_pix = tf.placeholder(tf.float32, [None, 784])
Weigh = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))
model = tf.matmul(in_image_pix, Weigh) + bias

print "\n----------------- Define loss and optimizer"
correct_answers = tf.placeholder(tf.float32, [None, 10])

#  cross_entropy = tf.reduce_mean(-tf.reduce_sum(correct_answers * tf.log(tf.nn.softmax(model)), reduction_indices=[1])) #numerically unstable
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=correct_answers, logits=model))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print "\n----------------- BEFORE TRAIN"

printData()

print "\n----------------- TRAINING"
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={in_image_pix: batch_xs, correct_answers: batch_ys})
    if i%100 == 0 or i < 10:
        print "\t\tTRAIN STEP: %d" % i
        testModelAndPrint()
        printErrorLos()

print "\n----------------- AFTER TRAIN"
printData()

writer = tf.summary.FileWriter("tmp/basic", sess.graph)
