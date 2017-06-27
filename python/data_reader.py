from __future__ import print_function

import time

import pickle
import random
import numpy as np

class DataReader(object):
    def __init__(self, is_training=True):

        ts = time.time()
        config = pickle.load(open('./embedded_data_config', 'rb'))
        self.vs = config['wordvec_size']
        self.max_postfix = config['postfix']
        self.cur_postfix = 0
        self.seq_max_length = config['seq_max_length']
        if is_training == True:
            print("\nLoading training data part", self.cur_postfix)
            self.train_X, self.train_Y, self.data_size = pickle.load(open('data/embedded_train_data_%d' % self.cur_postfix, "rb"))
            print("Training Data Size", self.data_size)
            self.index = random.sample(range(self.data_size), self.data_size)
            self.counter = 0
        else:
            print("\nLoading testing data...")
            self.test_X, self.test_Y, self.test_data_size = pickle.load(open('data/embedded_test_data', "rb"))
            self.testdata_index = 0
        print("Time Elapsed: {0} secs\n".format(time.time() - ts))

    def epoch_size(self, batch_size):
        return (self.data_size * self.max_postfix-1) // batch_size
    
    def validation_batch(self, batch_size):
        tmp_index = random.sample(range(self.data_size), self.data_size)
        batch_index = tmp_index[:batch_size]
        batch_x = [self.train_X[i] for i in batch_index]
        batch_y = [self.train_Y[i] for i in batch_index]
        return np.array(batch_x), np.array(batch_y)
    
    def load_next_training_data(self):
        ts = time.time()
        self.cur_postfix = (self.cur_postfix+1) % self.max_postfix
        print("\nLoading training data part", self.cur_postfix)
        self.train_X, self.train_Y, self.data_size = pickle.load(open('data/embedded_train_data_%d' % self.cur_postfix, "rb"))
        print("Time Elapsed: {0} secs\n".format(time.time() - ts))

    def generate_batch_index(self):
        self.counter = 0
        self.index = random.sample(range(self.data_size), self.data_size)
    
    def next_batch(self, batch_size):
        if self.counter + batch_size >= self.data_size:
            self.load_next_training_data()
            self.generate_batch_index()
        batch_index = self.index[self.counter:self.counter+batch_size]
        batch_x = [self.train_X[i] for i in batch_index]
        batch_y = [self.train_Y[i] for i in batch_index]
        self.counter += batch_size
        return np.array(batch_x), np.array(batch_y)

    def check_test_index(self):
        if self.testdata_index < self.test_data_size:
            return True
        else:
            return False

    def next_test(self, batch_size):
        if self.testdata_index < self.test_data_size:
            batch_x = [self.test_X[self.testdata_index+i] if i < 5 else np.zeros(shape=(self.seq_max_length, self.vs)) for i in range(batch_size)]
            batch_y = self.test_Y[self.testdata_index/5]
            self.testdata_index += 5
            return np.array(batch_x), np.array(batch_y)
        else:
            return []

    def new_next_test(self, batch_size):
        if self.testdata_index < self.test_data_size:
            batch_x = [self.test_X[self.testdata_index] if i == 0 else np.zeros(shape=(self.seq_max_length, self.vs)) for i in range(batch_size)]
            batch_y = [self.test_Y[self.testdata_index/5] if i == 0 else np.zeros(self.vs) for i in range(batch_size)]
            self.testdata_index += 1
            return np.array(batch_x), np.array(batch_y)
        else:
            return []

# class DataReader(object):
#     def __init__(self, is_training=True):

#         ts = time.time()
#         config = pickle.load(open('data/embedded_data_config', 'rb'))
#         self.vs = config['wordvec_size']
#         self.max_postfix = config['postfix']
#         self.cur_postfix = 0
#         self.seq_max_length = config['seq_max_length']
#         if is_training == True:
#             print("\nLoading training data part", self.cur_postfix)
#             self.train_X, self.train_Y, self.seqlen, self.data_size = pickle.load(open('data/embedded_train_data_%d' % self.cur_postfix, "rb"))
#             # for i in range(1, 6):
#             #     filename = 'data/embedded_train_data_%d' % i
#             #     train_X, train_Y, seqlen, data_size = pickle.load(open(filename, "rb"))
#             #     self.train_X = np.concatenate((self.train_X, train_X), axis=0)
#             #     self.train_Y = np.concatenate((self.train_Y, train_Y), axis=0)
#             #     self.seqlen = np.concatenate([self.seqlen, seqlen])
#             #     self.data_size += data_size
#             print("Training Data Seqlen Size", len(self.seqlen))
#             print("Training Data Size", self.data_size)
#             self.index = random.sample(range(self.data_size), self.data_size)
#             self.counter = 0
#         else:
#             print("\nLoading testing data...")
#             self.test_X, self.test_Y, self.test_seqlen, self.test_data_size = pickle.load(open('data/embedded_test_data', "rb"))
#             self.testdata_index = 0
#         print("Time Elapsed: {0} secs\n".format(time.time() - ts))

#     def epoch_size(self, batch_size):
#         return (self.data_size * self.max_postfix) // batch_size
    
#     def validation_batch(self, batch_size):
#         tmp_index = random.sample(range(self.data_size), self.data_size)
#         batch_index = tmp_index[:batch_size]
#         batch_x = [self.train_X[i] for i in batch_index]
#         batch_y = [self.train_Y[i] for i in batch_index]
#         seq = [self.seqlen[i] for i in batch_index]
#         return np.array(batch_x), np.array(batch_y), np.array(seq)
    
#     def load_next_training_data(self):
#         ts = time.time()
#         self.cur_postfix = (self.cur_postfix+1) % self.max_postfix
#         print("\nLoading training data part", self.cur_postfix)
#         self.train_X, self.train_Y, self.seqlen, self.data_size = pickle.load(open('data/embedded_train_data_%d' % self.cur_postfix, "rb"))
#         print("Time Elapsed: {0} secs\n".format(time.time() - ts))

#     def generate_batch_index(self):
#         self.counter = 0
#         self.index = random.sample(range(self.data_size), self.data_size)
    
#     def next_batch(self, batch_size):
#         if self.counter + batch_size >= self.data_size:
#             self.load_next_training_data()
#             self.generate_batch_index()
#         batch_index = self.index[self.counter:self.counter+batch_size]
#         batch_x = [self.train_X[i] for i in batch_index]
#         batch_y = [self.train_Y[i] for i in batch_index]
#         seq = [self.seqlen[i] for i in batch_index]
#         self.counter += batch_size
#         return np.array(batch_x), np.array(batch_y), np.array(seq)

#     def check_test_index(self):
#         if self.testdata_index < self.test_data_size:
#             return True
#         else:
#             return False

#     def next_test(self, batch_size):
#         if self.testdata_index < self.test_data_size:
#             batch_x = [self.test_X[self.testdata_index+i] if i < 5 else np.zeros(shape=(self.seq_max_length, self.vs)) for i in range(batch_size)]
#             batch_y = self.test_Y[self.testdata_index/5]
#             seq = [self.test_seqlen[self.testdata_index+i] if i < 5 else 0 for i in range(batch_size)]
#             self.testdata_index += 5
#             return np.array(batch_x), np.array(batch_y), np.array(seq)
#         else:
#             return []

#     def new_next_test(self, batch_size):
#         if self.testdata_index < self.test_data_size:
#             batch_x = [self.test_X[self.testdata_index] if i == 0 else np.zeros(shape=(self.seq_max_length, self.vs)) for i in range(batch_size)]
#             batch_y = [self.test_Y[self.testdata_index/5] if i == 0 else np.zeros(self.vs) for i in range(batch_size)]
#             # batch_y = self.test_Y[self.testdata_index/5]
#             seq = [self.test_seqlen[self.testdata_index] if i == 0 else 0 for i in range(batch_size)]
#             self.testdata_index += 1
#             return np.array(batch_x), np.array(batch_y), np.array(seq)
#         else:
#             return []