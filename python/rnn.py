from __future__ import print_function

import tensorflow as tf
# from tensorflow.contrib import rnn

import time

import numpy as np
from scipy import spatial

import pickle
import data_reader

config = pickle.load(open('./embedded_data_config', 'rb'))
# config['word_num'] = 38911
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


def RNN(x, weights, biases, is_training=True):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)

    # forget_bias needs to be fine-tuned.
    def lstm_cell(num_hidden):
        return tf.contrib.rnn.BasicLSTMCell(num_hidden, state_is_tuple=True)

    # wrapped_cell = lstm_cell

    # if is_training and keep_prob < 1:
    #     def wrapped_cell(num_hidden):
    #         return tf.contrib.rnn.DropoutWrapper(lstm_cell(num_hidden), output_keep_prob=keep_prob)

    # rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(n_hidden[i]) for i in range(num_layers)], state_is_tuple=True)

    # lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=0.9)
    # lstm_cell = rnn.DropoutWrapper(lstm_cell, output_keep_prob=0.9)
    # rnn_cell = rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)

    # One Layer RNN
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    outputs, _ = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    
    # Forward direction cell
    # lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)
    # Backward direction cell
    # lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, state_is_tuple=True)

    # Forward direction cell
    # lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(n_hidden[i]) for i in range(num_layers)], state_is_tuple=True)
    # Backward direction cell
    # lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(n_hidden[i]) for i in range(num_layers)], state_is_tuple=True)

    # Get lstm cell output
    # outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # outputs = []
    # state = rnn_cell.zero_state(batch_size, tf.float32)
    # with tf.variable_scope("RNN"):
    #     for time_step in range(n_steps):
    #         if time_step > 0: 
    #             tf.get_variable_scope().reuse_variables()
    #         (cell_output, state) = rnn_cell(x[time_step], state)
    #         outputs.append(cell_output)

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_hidden]
    # outputs = tf.stack(outputs)
    # outputs = tf.transpose(outputs, [1, 0, 2])

    # build the indexing and retrieve the right output
    # # index = [(i * n_steps + seqlen[i] - 2) for i in range(batch_size)]
    # index = tf.range(0, batch_size) * n_steps + (seqlen-1)
    # outputs = tf.reshape(outputs, [-1, n_hidden[-1]])
    # outputs = tf.gather(outputs, index)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

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
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(separate_cost)

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

# Launch the graph
with tf.Session() as sess:
# sess = tf.Session('', tf.Graph())
# with sess.graph.as_default():

    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # import data
    dr = data_reader.DataReader(is_training=True)
    valid_x, valid_y = dr.validation_batch(validation_size)
    valid_x = valid_x.reshape((validation_size, n_steps, n_input))
    valid_y = valid_y.reshape((validation_size, n_input))
    epoch_size = dr.epoch_size(batch_size)
    init_learning_rate = learning_rate

    # training
    for epoch in range(epoches):
        ts = time.time()
        learning_rate = init_learning_rate / (epoch+1)
        for batch_num in range(epoch_size):
            batch_x, batch_y = dr.next_batch(batch_size)
    
            # Reshape data to get n_steps seq of n_input elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            batch_y = batch_y.reshape((batch_size, n_input))
    
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            # s = sess.run(separate_cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seq})
            # c = sess.run(cost, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seq})
            # # s1 = sess.run(separate_cost_dim1, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seq})
            # print('s', s)
            # print('c', c)

            # p = sess.run(predict, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seq})
            # print("predictQuality", predictQuality(p, batch_y))
            if batch_num % display_step == 0:
                # Calculate batch accuracy
                correct_predict = []
                for i in range(validation_size / batch_size):
                    p = sess.run(predict, feed_dict={x: valid_x[i*batch_size:(i+1)*batch_size]})
                    correct_predict.append(predictQuality(p, valid_y[i*batch_size:(i+1)*batch_size]))
                correct_predict = reduce(lambda x, y: x + y, correct_predict) / len(correct_predict)
    
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Epoch:", epoch, "batch_num:", batch_num, "mini-batch Loss: " + "{:.6f}".format(loss) + ", Training Accuracy: " + "{:.5f}".format(correct_predict))
        print("Finish an epoch, time elapsed: {0} secs\n".format(time.time() - ts))
        # Save the session
        saver = tf.train.Saver()
        save_path = saver.save(sess, "tmp/model_layer1_epoch_%d.ckpt" % epoch)
        # saver.save(sess, 'model/rnn_one_layer_epoch_%d' % (epoch), meta_graph_suffix='meta', write_meta_graph=True)
    print("Optimization Finished!")

    saver = tf.train.Saver()
    # saver.save(sess, 'model/rnn_one_layer_epoch_end', meta_graph_suffix='meta', write_meta_graph=True)
    save_path = saver.save(sess, "tmp/model_layer1.ckpt")

