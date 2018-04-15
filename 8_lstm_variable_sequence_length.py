
# -*- coding: utf-8 -*-
import glob
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss

import numpy as np
from helpers import *
# ==============================================================================
nb_epoches = 15
learning_rate = 1.5

time_steps = 100  # number of inputs of RNN - 1, 
LEN_TEST_TEXT = 500 # Number of test characters of text to generate after training the network
#
batch_size = 1
chenckpoint_file = "tmp/LSTM_Model.ckpt"
pad_symbol="$"
# ================================ train data ==================================
TRAIN_DATA_FILE = "train_data/batch_text.txt"
input_data,batch_of_sentences = read_file_to_input_and_train_data(TRAIN_DATA_FILE,time_steps,64) #batch_size)
# ==============================================================================
print "\n----------------- BEFORE TRAIN"
batch_of_sentences = pad_strings(batch_of_sentences,pad_symbol)
number_of_train_samples = len(batch_of_sentences)

vocab, vocab_rev, num_classes = create_vocabulary_from_file(TRAIN_DATA_FILE)
hidden_size = num_classes

Train_data_batch = sentence_to_token_ids_from_batch(batch_of_sentences, vocab) #train data for loss calc
Input_data_one_hot_batch = data_array_to_one_hot_from_batch(sentence_to_token_ids_from_batch(input_data, vocab), vocab) #seed data

sequence_length = len(Train_data_batch[0]) # maximum length of sentences

print "batch_of_sentences"
print batch_of_sentences
print "Vocabulary"
print(vocab)
print "Train_data_batch"
print Train_data_batch
print "Input_data_one_hot_batch"
print_one_hot(Input_data_one_hot_batch)


# ==============================================================================
class LSTM_Model:
  def __init__(self, hidden_size, batch_size, num_classes, sequence_length):
    self.hidden_size = hidden_size
    self.batch_size = batch_size
    self.num_classes = num_classes
    self.sequence_length = sequence_length

    self.Input_data = tf.placeholder(tf.float32, [None, sequence_length, hidden_size],name="Input_data") # X onehot
    self.Train_data = tf.placeholder(tf.int32, [None, sequence_length],name="Train_data") #Y

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

    X_for_fc = tf.reshape(outputs_d_rnn, [-1, hidden_size])
    outputs_fc = fully_connected(inputs=X_for_fc,
                              num_outputs=num_classes,
                              activation_fn=None)
    outputs = tf.reshape(outputs_fc, [batch_size, sequence_length, num_classes])
    return outputs

    # outputs_d_rnn = tf.Print(outputs_d_rnn,[outputs_d_rnn],"\n--PRINT-- outputs_d_rnn:\n",summarize=1000)
    # return outputs_d_rnn

  @define_scope
  def prediction(self):
    prediction = tf.argmax(self._outputs, axis=2)
    return prediction

  @define_scope
  def loss(self):
    outputs = self._outputs
    train_data = self.Train_data
    weights = tf.ones([batch_size, sequence_length])

    seq = sequence_loss(logits=outputs,       #predictions
                        targets=train_data,   #true data
                        weights=weights)
    loss = tf.reduce_mean(seq)
    tf.summary.scalar('loss', loss) #log loss
    return loss

  @define_scope
  def train(self):
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
    return train

# ==============================================================================
with tf.Session() as sess:
  model = LSTM_Model(hidden_size, batch_size, num_classes, sequence_length)
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()

  print "\n----------------- TRAINING"
  if glob.glob(chenckpoint_file + "*"):
    modelPath = saver.restore(sess, chenckpoint_file)
    print "\n----------------- Skip training. Loaded from: " + chenckpoint_file
    model_was_trained = False
  else:
    tensorboard_logs_writer = tf.summary.FileWriter("tmp/basic")
    tensorboard_merged = tf.summary.merge_all()
    for i in range(nb_epoches):
      #--------training---------
      for j in range(len(Train_data_batch)):
        input_data_one_hot = [Input_data_one_hot_batch[j]]
        train_data_item = [Train_data_batch[j]]
        sess.run(model.train, feed_dict={model.Input_data: input_data_one_hot, model.Train_data: train_data_item}) 
      #--------------------------
      if i % (nb_epoches/100.0) == 0: #log loss
        tensorboard_summary, l = sess.run([tensorboard_merged,model.loss], feed_dict={model.Input_data: input_data_one_hot, model.Train_data: train_data_item })
        tensorboard_logs_writer.add_summary(tensorboard_summary, i)
      if i % (nb_epoches/10.0) == 0: #print model state
        l, result = sess.run([ model.loss, model.prediction], feed_dict={model.Input_data: input_data_one_hot, model.Train_data: train_data_item })
        print "epoch: %3d/%3d loss: %2.6f prediction: %s first true Y: %s" % (i,nb_epoches,l,result,train_data_item[0])

    model_was_trained = True

  print "\n----------------- AFTER TRAIN"
  if model_was_trained:
    writer = tf.summary.FileWriter("tmp/basic", sess.graph) # save model graph
    save_path = saver.save(sess, chenckpoint_file)
    print("Model saved in file: %s" % save_path)

  print "Generating string:\n"
  string_token_ids = one_hot_batch_to_array(Input_data_one_hot_batch[0]) 
  for i in range(LEN_TEST_TEXT):
    if i % (LEN_TEST_TEXT/10.0) == 0:
      print "Generating string: %3d/%3d" % (i,LEN_TEST_TEXT)
    history_data_one_hot = string_token_ids[-time_steps:]
    input_data_one_hot = [array_to_one_hot_batch(history_data_one_hot,len(vocab))]
    new_string_token_ids = sess.run(model.prediction, feed_dict={model.Input_data: input_data_one_hot})
    string_token_ids.append(new_string_token_ids[0][-1]) #last item - it is a generated symbol id

  print token_ids_to_sentence(string_token_ids,vocab_rev).replace(pad_symbol, "")