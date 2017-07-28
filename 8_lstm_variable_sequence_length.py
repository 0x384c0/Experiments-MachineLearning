
# -*- coding: utf-8 -*-
import glob
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss

import numpy as np
from helpers import *
# ==============================================================================
chenckpoint_file = "tmp/LSTM_Model.ckpt"
nb_epoches = 50
batch_size = 1
batch_of_sentences = [
"- string 68",
"- string 2388",
"- string 62323",
"- string 6",
"- string 66468",
"- string 538",
]
pad_symbol="$"
# ==============================================================================
print "\n----------------- BEFORE TRAIN"
batch_of_sentences = pad_strings(batch_of_sentences,pad_symbol)
number_of_train_samples = len(batch_of_sentences)

vocab, vocab_rev = create_vocabulary_from_batch(batch_of_sentences)
num_classes = len(vocab) # number of unique ids
hidden_size = num_classes

#train data
Y_data = sentence_to_token_ids_from_batch(batch_of_sentences, vocab) #train data for loss calc

#seed data
X_data = sentence_to_token_ids(batch_of_sentences[0], vocab)
sequence_length = len(X_data) # FIXED LENGHT
X_data_one_hot = [np.zeros((sequence_length, num_classes))] # seed data
X_data_one_hot[0][0][(X_data[0])] = 1

print "batch_of_sentences"
print batch_of_sentences
print "Vocabulary"
print(vocab)
print "Train Data"
print Y_data
print "X Data one hot (Seed data i guess)"
print_one_hot(X_data_one_hot)


# ==============================================================================
class LSTM_Model:
  def __init__(self, hidden_size, batch_size, num_classes, sequence_length):
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.sequence_length = sequence_length

    self.X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size]) # X onehot
    self.Y = tf.placeholder(tf.int32, [None, sequence_length])

    self.prediction
    self.train


  @define_scope
  def _outputs(self):


    cell = BasicLSTMCell(num_units=hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs_d_rnn, _states = tf.nn.dynamic_rnn(cell,
                                         self.X,
                                         initial_state=initial_state,
                                         dtype=tf.float32)

    X_for_fc = tf.reshape(outputs_d_rnn, [-1, hidden_size])
    outputs_fc = fully_connected(inputs=X_for_fc,
                              num_outputs=num_classes,
                              activation_fn=None)
    outputs = tf.reshape(outputs_fc, [batch_size, sequence_length, num_classes])
    return outputs

  @define_scope
  def prediction(self):
    prediction = tf.argmax(self._outputs, axis=2)
    return prediction

  @define_scope
  def loss(self):
    weights = tf.ones([batch_size, sequence_length])
    seq = sequence_loss(logits=self._outputs,
                                  targets=self.Y,
                                  weights=weights)
    loss = tf.reduce_mean(seq)
    return loss

  @define_scope
  def train(self):
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
    return train
# ==============================================================================
with tf.Session() as sess:
  model = LSTM_Model(hidden_size, batch_size, num_classes, sequence_length)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  print "\n----------------- TRAINING"
  if glob.glob(chenckpoint_file + "*"):
    model_was_trained = False
    modelPath = saver.restore(sess, chenckpoint_file)
    print "\n----------------- Skip training. Loaded from: " + chenckpoint_file
  else:
    model_was_trained = True
    for i in range(nb_epoches):
      for item in Y_data:
        item = [item]
        sess.run(model.train, feed_dict={model.X: X_data_one_hot, model.Y: item}) 

      if i % (nb_epoches/10) == 0:
        l, result = sess.run([model.loss, model.prediction], feed_dict={model.X: X_data_one_hot, model.Y: item })
        print "step: %3d loss: %2.6f prediction: %s first true Y: %s" % (i,l,result,Y_data[0])

  print "\n----------------- AFTER TRAIN"
  if model_was_trained:
    writer = tf.summary.FileWriter("tmp/basic", sess.graph)
    save_path = saver.save(sess, chenckpoint_file)
    print("Model saved in file: %s" % save_path)

  print "\nResult:"
  result = sess.run(model.prediction, feed_dict={model.X: X_data_one_hot, })
  print token_ids_to_sentence(result,vocab_rev).replace(pad_symbol, "")