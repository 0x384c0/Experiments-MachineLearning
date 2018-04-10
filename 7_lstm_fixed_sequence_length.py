
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.layers import fully_connected
from tensorflow.contrib.seq2seq import sequence_loss

import numpy as np
from helpers import *
# ==============================================================================

print "\n----------------- BEFORE TRAIN"
nb_epoches = 50
# train data  ------------------------------------------------------------------------------------batch_of_sentences = [
# batch_of_sentences = ["- string 2323458"]
# batch_of_sentences = [
# "- string 23$$$",
# "- string 42$$$",
# "- string 52323",
# "- string 0$$$$",
# "- string 86464",
# "- string 933$$",
# ]
batch_of_sentences = read_file_to_array_of_strings("train_data/batch_circle.txt")
# ----------------------------------------------------------------------------------------
batch_size = len(batch_of_sentences)
if batch_size == 1:
# single ------------------------------------------------------------------------------------
    train_sentence = batch_of_sentences[0]
    vocab, vocab_rev = create_vocabulary(train_sentence)
    X_data = sentence_to_token_ids(train_sentence, vocab)
    num_classes = len(vocab) # number of unique ids
    hidden_size = num_classes
    sequence_length = len(X_data) # FIXED LENGHT

    X_data_one_hot = [data_array_to_one_hot(X_data, vocab)]
    Y_data = [X_data] #train data for loss calc
else:
# batch -------------------------------------------------------------------------------------------
    vocab, vocab_rev = create_vocabulary_from_batch(batch_of_sentences)
    X_data = sentence_to_token_ids_from_batch(batch_of_sentences, vocab)
    num_classes = len(vocab) # number of unique ids
    hidden_size = num_classes
    sequence_length = len(X_data[0]) # FIXED LENGHT

    X_data_one_hot = data_array_to_one_hot_from_batch(X_data, vocab)
    Y_data = X_data #train data for loss calc
# ----------------------------------------------------------------------------------------



print "Vocabulary"
print(vocab)
print "Train Data"
print Y_data
print "X Data one hot (Seed data i guess)"
print_one_hot(X_data_one_hot)


# ==============================================================================

X = tf.placeholder(tf.float32, [None, sequence_length, hidden_size]) # X onehot
Y = tf.placeholder(tf.int32, [None, sequence_length])


cell = BasicLSTMCell(num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs_d_rnn, _states = tf.nn.dynamic_rnn(cell,
                                     X,
                                     initial_state=initial_state,
                                     dtype=tf.float32)

X_for_fc = tf.reshape(outputs_d_rnn, [-1, hidden_size])
outputs_fc = fully_connected(inputs=X_for_fc,
                          num_outputs=num_classes,
                          activation_fn=None)

outputs = tf.reshape(outputs_fc, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])
sequence_loss = sequence_loss(logits=outputs,
                              targets=Y,
                              weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)
# ==============================================================================


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    X_in = X_data_one_hot
    X_in_seed = X_in#[np.zeros((sequence_length, num_classes))] # seed data
    # X_in_seed[0][0][(X_data[0])] = 1
    print "\n----------------- TRAINING"

    for i in range(nb_epoches):
        l, _ = sess.run([loss, train], feed_dict={X: X_in, Y: Y_data}) 
        if i % (nb_epoches/10) == 0:
            result = sess.run(prediction, feed_dict={X: X_in_seed, })
            print "step: %3d loss: %2.6f prediction: %s true Y: %s" % (i,l,result,Y_data)
            print token_ids_to_sentence(result,vocab_rev)

    print "\n----------------- AFTER TRAIN"
    # step by step model eval


    outputs_d_rnn_out   = sess.run(outputs_d_rnn,   { X             : X_in_seed                         })
    X_for_fc_out        = sess.run(X_for_fc,        { outputs_d_rnn : outputs_d_rnn_out                 })
    outputs_fc_out      = sess.run(outputs_fc,      { X_for_fc      : X_for_fc_out                      })
    outputs_out         = sess.run(outputs,         { outputs_fc    : outputs_fc_out                    })
    prediction_out      = sess.run(prediction,      { outputs       : outputs_out                       })

    sequence_loss_out   = sess.run(sequence_loss,   { outputs       : outputs_out,          Y : Y_data  })
    loss_out            = sess.run(loss,            { sequence_loss : sequence_loss_out                 })

    # print "\noutputs_d_rnn_out"
    # print outputs_d_rnn_out

    # print "\nX_for_fc_out"
    # print X_for_fc_out

    # print "\noutputs_fc_out"
    # print outputs_fc_out

    # print "\nsequence_loss_out"
    # print sequence_loss_out

    # print "\nloss_out"
    # print loss_out

    print "\nPredicted Data one hot"
    print_one_hot(outputs_out)

    print "\nResult:"
    print token_ids_to_sentence(prediction_out,vocab_rev)