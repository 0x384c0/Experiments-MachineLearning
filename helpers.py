import tensorflow as tf
import numpy as np
import random

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

def data_array_to_one_hot(data_array, vocab_array):
  token_ids_one_hot = np.zeros((len(data_array), len_of_vocab(vocab_array)))
  token_ids_one_hot[np.arange(len(data_array)), data_array] = 1
  return token_ids_one_hot


def data_array_to_one_hot_from_batch(batch, vocab_array):
  batch_of_token_ids_one_hot = []
  for data_array in batch:
    batch_of_token_ids_one_hot.append(data_array_to_one_hot(data_array,vocab_array))
  return batch_of_token_ids_one_hot

def one_hot_batch_to_array(one_hot_batch):
  array = []
  for one_hot in one_hot_batch:
    array.append(np.argmax(one_hot))
  return array

def array_to_one_hot_batch(array,max_value):
  one_hot_batch = np.zeros((len(array), max_value+1))
  one_hot_batch[np.arange(len(array)),array] = 1
  return one_hot_batch



# vocabulary
vocab_start_id = 1
def len_of_vocab(vocab):
  return len(vocab) + vocab_start_id

def create_vocabulary(string):
  vocab = {}
  char_id = vocab_start_id
  for char in string:
    if char not in vocab:
      vocab[char] = char_id
      char_id += 1

  vocab_rev = {v: k for k, v in vocab.iteritems()}

  return vocab, vocab_rev, len_of_vocab(vocab)

def create_vocabulary_from_batch(batch):
  vocab = {}
  char_id = vocab_start_id
  for string in batch:
    for char in string:
      if char not in vocab:
        vocab[char] = char_id
        char_id += 1

  vocab_rev = {v: k for k, v in vocab.iteritems()}

  return vocab, vocab_rev, len_of_vocab(vocab)

def create_vocabulary_from_file(file_name):
  string = ""
  with open(file_name, 'r') as f:
    string += f.read()
  return create_vocabulary(string)


def sentence_to_token_ids(sentence, vocabulary):
  characters = [sentence[i:i+1] for i in range(0, len(sentence), 1)]
  return [vocabulary.get(w) for w in characters]
    
def sentence_to_token_ids_from_batch(batch, vocabulary):
  characters_batch = []
  for sentence in batch:
    characters_batch.append(sentence_to_token_ids(sentence,vocabulary))
  return characters_batch

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

# files

def read_file_to_array_of_strings(file_name):
  return [line.rstrip('\n') for line in open(file_name)]

def read_files_to_array_of_strings(file_names):
  strings = []
  for file_name in file_names:
    strings.append(open(file_name).read())
  return strings

def read_file_to_input_and_train_data(file_name,time_steps,batch_size):
  input_data = []
  train_data = []
  string = ""
  with open(file_name, 'r') as f:
    string += f.read()

  if len(string) < time_steps + 2:
    raise ValueError("Lenght of \"" + file_name + "\" file (" + str(len(string)) + ") must be greater then " + str(time_steps + 2))

  possible_batch_ids = range(len(string) - time_steps - 1)
  for i in range(batch_size):
    batch_id = random.choice(possible_batch_ids)
    input_data.append(string[batch_id:batch_id + time_steps])
    train_data.append(string[batch_id + 1:batch_id + time_steps + 1])

  return input_data,train_data




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
