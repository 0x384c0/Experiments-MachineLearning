import tensorflow as tf
import numpy as np

# Weight Initialization function
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
# Convolution and Pooling
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')



# ============================================================================== rnn
def print_one_hot(one_hot):
    for array in one_hot[0]:
        # print(" ".join(map(lambda value: '%f' % value, array)),"  max_value ",max(array),"  max_value_index", array.argmax())
        print "%s \tmax_value: %f, \tmax_value_index: %d" % ("\t".join(map(lambda value: "% 2.2f" % (value), array)),max(array),array.argmax())

def create_vocabulary(sequence):
    vocab = {}
    for i in range(len(sequence)):
        ch = sequence[i]
        if ch in vocab:
            vocab[ch] += 1
        else:
            vocab[ch] = 1
    vocab_rev = sorted(vocab, key=vocab.get, reverse=True)
    vocab = dict([(x, y) for (y, x) in enumerate(vocab_rev)])
    return vocab, vocab_rev


def sentence_to_token_ids(sentence, vocabulary):
    characters = [sentence[i:i+1] for i in range(0, len(sentence), 1)]
    return [vocabulary.get(w) for w in characters]

def token_ids_to_sentence(ids, vocabulary_rev):
    return ''.join([vocabulary_rev[c] for c in np.squeeze(ids)])


def data_array_to_one_hot(data_array, vocab_array):
    token_ids_one_hot = np.zeros((len(data_array), len(vocab_array)))
    token_ids_one_hot[np.arange(len(data_array)), data_array] = 1
    return token_ids_one_hot
# ==============================================================================