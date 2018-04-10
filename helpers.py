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
def pad_strings(strings,pad_symbol="$"):
  result = []
  max_len = 0
  for string in strings:
    if max_len < len(string):
      max_len = len(string)

  for string in strings:
    result.append(string + (pad_symbol * (max_len - len(string))))
  return result


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
    ids_squeezed = np.squeeze(ids)
    if isinstance(ids_squeezed[0], np.ndarray):
      result = ''
      for ids_arr in ids:
        result += "\n"
        result += ''.join([vocabulary_rev[c] for c in ids_arr])
      return result
    else:
      return ''.join([vocabulary_rev[c] for c in ids_squeezed])


def data_array_to_one_hot(data_array, vocab_array):
    token_ids_one_hot = np.zeros((len(data_array), len(vocab_array)))
    token_ids_one_hot[np.arange(len(data_array)), data_array] = 1
    return token_ids_one_hot



def create_vocabulary_from_batch(batch):
    vocab = {}
    for i in range(len(batch)):
        sequence = batch[i]
        for i in range(len(sequence)):
            ch = sequence[i]
            vocab[ch] = 1
            # if ch in vocab:
            #     vocab[ch] += 1
            # else:
            #     vocab[ch] = 1
    vocab_rev = sorted(vocab, key=vocab.get, reverse=True)
    vocab = dict([(x, y) for (y, x) in enumerate(vocab_rev)])
    return vocab, vocab_rev


def sentence_to_token_ids_from_batch(batch, vocabulary):
    characters_batch = []
    for sentence in batch:
      characters_batch.append(sentence_to_token_ids(sentence,vocabulary))
    return characters_batch

def data_array_to_one_hot_from_batch(batch, vocab_array):
    batch_of_token_ids_one_hot = []
    for data_array in batch:
      batch_of_token_ids_one_hot.append(data_array_to_one_hot(data_array,vocab_array))
    return batch_of_token_ids_one_hot

def read_file_to_array_of_lines(file_name):
  return [line.rstrip('\n') for line in open(file_name)]

def read_files_to_array_of_strings(file_names):
  strings = []
  for file_name in file_names:
    strings.append(open(file_name).read())
  return strings

# Lazy Property Decorator
import functools
def define_scope(function):
    attribute = '_cache_' + function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
# ==============================================================================