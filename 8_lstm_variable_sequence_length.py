
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
nb_epoches = 50 * 30
batch_size = 1
learning_rate = 0.1 / 30
pad_symbol="$"
# ================================ train data ==================================
batch_of_sentences = [
"- string 68",
"- string 2388",
"- string 62323",
"- string 6",
"- string 66468",
"+ rsting 538",
"+ rsting 528",
"+ rsting 338",
"+ rsting 638",
"+ rsting 228",
]
# batch_of_sentences = read_files_to_array_of_strings([
#   "train_data/batch_circle.txt",
#   "train_data/batch_circle.txt",
#   "train_data/batch_circle.txt",
#   "train_data/batch_cross.txt",
#   "train_data/batch_cross.txt",
#   "train_data/batch_cross.txt",
#   ])
# ==============================================================================
print "\n----------------- BEFORE TRAIN"
batch_of_sentences = pad_strings(batch_of_sentences,pad_symbol)
number_of_train_samples = len(batch_of_sentences)

vocab, vocab_rev = create_vocabulary_from_batch(batch_of_sentences)
num_classes = len(vocab) # number of unique ids
hidden_size = num_classes

#train data
train_data = sentence_to_token_ids_from_batch(batch_of_sentences, vocab) #train data for loss calc

#seed data
input_data = sentence_to_token_ids(batch_of_sentences[0], vocab) #take first sentence
sequence_length = len(input_data) # maximum length of sentences
Input_data_one_hot = [np.zeros((sequence_length, num_classes))] # seed data
Input_data_one_hot[0][0][(input_data[0])] = 1 #first symbol of first sentence

print "batch_of_sentences"
print batch_of_sentences
print "Vocabulary"
print(vocab)
print "Train Data"
print train_data
print "Input_data_one_hot"
print_one_hot(Input_data_one_hot)


# ==============================================================================
class LSTM_Model:
  def __init__(self, hidden_size, batch_size, num_classes, sequence_length):
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.sequence_length = sequence_length

    self.Input_data = tf.placeholder(tf.float32, [None, sequence_length, hidden_size]) # Input_data onehot
    self.Train_data = tf.placeholder(tf.int32, [None, sequence_length])

    self.prediction
    self.train


  @define_scope
  def _outputs(self):
    cell = BasicLSTMCell(num_units=hidden_size)
    initial_state = cell.zero_state(batch_size, tf.float32)
    outputs_d_rnn, _states = tf.nn.dynamic_rnn(cell,
                                         self.Input_data,
                                         initial_state=initial_state,
                                         dtype=tf.float32)
    # outputs_d_rnn = tf.Print(outputs_d_rnn,[outputs_d_rnn],"\n--PRINT-- outputs_d_rnn:\n",summarize=1000)
    return outputs_d_rnn

  @define_scope
  def prediction(self):
    prediction = tf.argmax(self._outputs, axis=2)
    return prediction

  @define_scope
  def loss(self):
    weights = tf.ones([batch_size, sequence_length])
    outputs = self._outputs
    train_data = self.Train_data
    input_data = self.Input_data


    seq = sequence_loss(logits=outputs, #predictions
                        targets=train_data,       #true data
                        weights=weights)
    loss = tf.reduce_mean(seq)
    return loss

  @define_scope
  def train(self):
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
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
      for item in train_data:
        item = [item]
        sess.run(model.train, feed_dict={model.Input_data: Input_data_one_hot, model.Train_data: item}) 

      if i % (nb_epoches/10) == 0:
        l, result = sess.run([model.loss, model.prediction], feed_dict={model.Input_data: Input_data_one_hot, model.Train_data: item })
        print "step: %3d loss: %2.6f prediction: %s first true Y: %s" % (i,l,result,train_data[0])

  print "\n----------------- AFTER TRAIN"
  if model_was_trained:
    writer = tf.summary.FileWriter("tmp/basic", sess.graph)
    save_path = saver.save(sess, chenckpoint_file)
    print("Model saved in file: %s" % save_path)

  print "\nResult:"
  result = sess.run(model.prediction, feed_dict={model.Input_data: Input_data_one_hot, })
  print token_ids_to_sentence(result,vocab_rev).replace(pad_symbol, "")