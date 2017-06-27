from __future__ import print_function

import tensorflow as tf
# from tensorflow.contrib import rnn

import re

import os
import sys

import time
import pickle

import numpy as np
from scipy import spatial
from gensim.models import word2vec, KeyedVectors

import pickle
import data_reader

pretrained_model = 'tmp/model_layer1.ckpt'
config = pickle.load(open('./embedded_data_config', 'rb'))
print('config', config)

# Training Parameters
learning_rate = 0.000018
epoches = 2
batch_size = 256
display_step = 100
validation_size = batch_size * 10

# Network Parameters
num_layers = 1                              # RNN cell layers
n_hidden = 800                              # hidden layer num of features
keep_prob = 0.9                             # rnn layer output dropout
n_steps = config['seq_max_length']          # timesteps (depend on the sentence's length)
n_input = config['wordvec_size']            # data input
n_classes = config['wordvec_size']          # output vector

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
# seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]), name="w")
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]), name="b")
}

""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data

def RNN(x, weights, biases, is_training=True):
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # One Layer RNN
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def predictQuality(list1, list2):
    threshold_prob = 0.4  # two vector's angel < 60 degrees
    pred = np.array([1 if 1 - spatial.distance.cosine(list1[i], list2[i]) >= threshold_prob else 0 for i in range(batch_size)])
    return sum(pred) / float(len(pred))

def test_predictAnswer(vector_list, test_y):
    answer_list = ['a', 'b', 'c', 'd', 'e']
    max_cosineSimilarity = -float("inf")
    max_index = -1
    for i in range(5):
        cosineSimilarity = 1 - spatial.distance.cosine(vector_list[i], test_y)
        # print("cosine", answer_list[i], cosineSimilarity)
        if cosineSimilarity > max_cosineSimilarity:
            max_cosineSimilarity = cosineSimilarity
            max_index = i
    return answer_list[max_index]

predict = RNN(x, weights, biases)

# with tf.variable_scope("RNN") as scope:
#     predict = RNN(x, seqlen, weights, biases)
#     scope.reuse_variables()
#     test_predict = RNN(x, seqlen, weights, biases, is_training=False)

# Define loss and optimizer
# separate_cost = tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y)
# separate_cost = tf.nn.nce_loss(weights, biases, y, predict, config['word_num']/2, config['word_num'])
separate_cost = tf.losses.cosine_distance(y, predict, dim=1)
# separate_cost_dim1 = tf.losses.cosine_distance(y, predict, dim=1)
cost = tf.reduce_mean(separate_cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(separate_cost)

# parse testing data
print("Parsing testing data...")
ts = time.time()
testing_data = []
with open(sys.argv[1], 'r') as f:
    next(f)  # ignore the first line
    for line in f:
        line = line.split(',')
        line[-1] = line[-1][:-1]  # remove '\r\n' at the end of the line
        question = " ".join(line[1:-5]).lower().split()
        question = [refine(word) if word != '_____' else '_____' for word in question]
        testing_data.append({
            'id': line[0],
            'question': question,
            'a': refine(line[-5].lower()),
            'b': refine(line[-4].lower()),
            'c': refine(line[-3].lower()),
            'd': refine(line[-2].lower()),
            'e': refine(line[-1].lower())
        })
        line = refine(" ".join(line[1:]).lower())
pickle.dump(testing_data, open('data/testing_data', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))

WORD_VECTOR_SIZE = 300
MIN_WORD_COUNT = 20
MAX_SENTENCE_LENGTH = 15
MIN_SENTENCE_LENGTH = 8
training_data_size = 100000

word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

# find some attribute in testdata
print("Loading raw testing data...")
testing_data = pickle.load(open('data/testing_data', 'rb'))
seqlen = []
embedded_test_X = []
embedded_test_Y = []
onehot_test_X = []
onehot_test_Y = []
max_test_length = 0
min_test_length = float("inf")
max_space_index = 0
min_space_index = float("inf")
sum_length = 0
sum_space_index = 0
for test_case in testing_data:
    sum_length += len(test_case['question'])
    if len(test_case['question']) > max_test_length:
        max_test_length = len(test_case['question'])
    if len(test_case['question']) < min_test_length:
        min_test_length = len(test_case['question'])
    index = test_case['question'].index('_____')
    sum_space_index += index
    if index > max_space_index:
        max_space_index = index
    if index < min_space_index:
        min_space_index = index
average_length = sum_length / float(len(testing_data))
average_space_index = sum_space_index / float(len(testing_data))
seq_max_length = max_space_index + 1
print("max_test_length", max_test_length)
print("min_test_length", min_test_length)
print("average_length", average_length)
print("max_space_index", max_space_index)
print("min_space_index", min_space_index)
print("average_space_index", average_space_index)
print("seq_max_length", seq_max_length)

# parse testing dats
sw = open('model/stopwords.txt', 'r').read().split('\n')
embedded_context_test = []
for test_case in testing_data:
    index = test_case['question'].index('_____')

    # pre-context
    preContextVector = np.array([word_vector[word] if word not in sw and word in word_vector and word != '' else np.zeros(WORD_VECTOR_SIZE) for word in test_case['question'][:index]])
    preContextVector = np.sum(preContextVector, axis=0)

    # post-context
    postContextVector = np.array([word_vector[word] if word not in sw and word in word_vector and word != '' else np.zeros(WORD_VECTOR_SIZE) for word in test_case['question'][index+1:]])
    postContextVector = np.sum(postContextVector, axis=0)

    for key in ['a', 'b', 'c', 'd', 'e']:
        if test_case[key] in word_vector:
            midContextVector = word_vector[test_case[key]]
        else:
            midContextVector = np.zeros(WORD_VECTOR_SIZE)
        embedded_test_X.append([preContextVector, midContextVector])
    embedded_test_Y.append(postContextVector)
pickle.dump((np.array(embedded_test_X), np.array(embedded_test_Y), len(embedded_test_X)), open('data/embedded_test_data', 'wb'), True)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, pretrained_model)
    print("Model restored.")
    
    # testing
    ts = time.time()
    dr = data_reader.DataReader(is_training=False)
    with open(sys.argv[2], 'w') as out:
        out.write('id,answer\n')
        test_case = 1
        while dr.check_test_index() == True:
            test_x, test_y = dr.next_test(batch_size)

            tp = sess.run(predict, feed_dict={x: test_x})
            predict_answer = test_predictAnswer(tp[0:5], test_y)

            out.write("%s,%s\n" % (test_case, predict_answer))
            test_case += 1
