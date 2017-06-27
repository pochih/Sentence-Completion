from __future__ import print_function

import pickle
import time

import numpy as np

from collections import Counter
from gensim.models import word2vec, KeyedVectors


WORD_VECTOR_SIZE = 300
MIN_WORD_COUNT = 20
MAX_SENTENCE_LENGTH = 15
MIN_SENTENCE_LENGTH = 8
training_data_size = 100000


# build word dictionary  
print("Building word dictionary...")
ts = time.time()
word_dict = Counter(open("model/all_words.txt", 'r').read().split())
testing_data_word_dict = Counter(open("model/testing_data_words.txt", 'r').read().split())
# print("Top 20 most appearance", word_dict.most_common(20))
for word in list(word_dict):
    if word_dict[word] < MIN_WORD_COUNT and word not in testing_data_word_dict:
        del word_dict[word]
print("word_dict size", len(word_dict.keys()))
print("testing_data_word_dict size", len(testing_data_word_dict.keys()))
pickle.dump(word_dict, open('model/word_dict', 'wb'), True)
pickle.dump(testing_data_word_dict, open('model/testing_data_word_dict', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# build word vector, type: one-hot encoding
""" Mention that every time you run the code,
    words with same count will have different orders.
    e.g. every time you run the code,
         'the' will always be the first index,
         while 'nervous' may have different order.
"""
print("Building one-hot word vector...")
ts = time.time()
word_list = word_dict.most_common()
one_hot_dict = {}
for i in range(len(word_list)):
    one_hot_dict[word_list[i][0]] = i
pickle.dump(one_hot_dict, open('model/one_hot_dict', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# build word vector, type: word embedding
# print("Building embedded word vector...")
# ts = time.time()
# corpus = word2vec.Text8Corpus("model/all_words.txt")
# word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
# word_vector.wv.save_word2vec_format(u"model/word_vector.txt", binary=False)
# word_vector.wv.save_word2vec_format(u"model/word_vector.bin", binary=True)
# print("Time Elapsed: {0} secs\n".format(time.time() - ts))


''' apply two types of vector to training data '''
# loading necessary dictionaries
word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)
word_dict = pickle.load(open('model/word_dict', 'rb'))
one_hot_dict = pickle.load(open('model/one_hot_dict', 'rb'))

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


# parse training data
ts = time.time()
print("Loading raw training data...")
sentences = pickle.load(open('data/training_data', 'rb'))
embedded_train_X = []
embedded_train_Y = []
counter = 0
postfix = -1
for sentence in sentences:
    if max_test_length >= len(sentence) >= min_test_length:
        valid_sentence_flag = 1
        for word in sentence:
            if word not in word_vector:
                valid_sentence_flag = 0
                break
        if valid_sentence_flag == 1:
            mid_index = len(sentence)/2

            # pre-context
            preContextVector = np.array([word_vector[word] if word not in sw and word in word_vector and word != '' else np.zeros(WORD_VECTOR_SIZE) for word in sentence[:mid_index]])
            preContextVector = np.sum(preContextVector, axis=0)

            # post-context
            postContextVector = np.array([word_vector[word] if word not in sw and word in word_vector and word != '' else np.zeros(WORD_VECTOR_SIZE) for word in sentence[mid_index+1:]])
            postContextVector = np.sum(postContextVector, axis=0)

            if sentence[mid_index] in word_vector:
                midContextVector = word_vector[sentence[mid_index]]
            else:
                midContextVector = np.zeros(WORD_VECTOR_SIZE)

            embedded_train_X.append([preContextVector, midContextVector])
            embedded_train_Y.append(postContextVector)

            counter += 1
            if counter % training_data_size == (training_data_size-1):
                postfix += 1
                print("Save file, postfix:", postfix)
                pickle.dump((np.array(embedded_train_X), np.array(embedded_train_Y), len(embedded_train_X)), open('data/embedded_train_data_%d' % (postfix), 'wb'), True)
                embedded_train_X = []
                embedded_train_Y = []
pickle.dump({
        'postfix': postfix,
        'word_num': len(word_dict.keys()),
        'seq_max_length': 2,
        'max_test_length': max_test_length,
        'min_test_length': min_test_length,
        'max_space_index': max_space_index,
        'min_space_index': min_space_index,
        'onehot_vec_size': len(word_dict.keys()),
        'wordvec_size': WORD_VECTOR_SIZE
    }, open('./embedded_data_config', 'wb'), True)
print("len(embedded_train_X)", len(embedded_train_X))
print("len(embedded_train_Y)", len(embedded_train_Y))
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# for test_case in testing_data:
#     index = test_case['question'].index('_____')
#     for key in ['a', 'b', 'c', 'd', 'e']:
#         test_case['question'][index] = test_case[key]
#         sentence = test_case['question']
#         seqlen.append(index+1)
#         embedded_test_X.append([word_vector[sentence[i]] if i <= index and sentence[i] in word_vector else np.zeros(WORD_VECTOR_SIZE) for i in range(seq_max_length)])
#         onehot_test_X.append([one_hot_dict[sentence[i]] if i <= index and sentence[i] in word_dict else -1 for i in range(seq_max_length)])
#     if sentence[index+1] != '':
#         embedded_test_Y.append(word_vector[sentence[index+1]])
#         onehot_test_Y.append(one_hot_dict[sentence[index+1]])
#     else:
#         embedded_test_Y.append(np.zeros(WORD_VECTOR_SIZE))
#         onehot_test_Y.append(-1)
# pickle.dump((np.array(embedded_test_X), np.array(embedded_test_Y), np.array(seqlen), len(embedded_test_X)), open('data/embedded_test_data', 'wb'), True)
# pickle.dump((np.array(onehot_test_X), np.array(onehot_test_Y), np.array(seqlen), len(onehot_test_X)), open('data/onehot_test_data', 'wb'), True)

# # word_count = len(word_dict.keys()) / 2
# # print("word count median", word_dict.most_common()[word_count-5:word_count+5])
# print("Loading raw training data...")
# ts = time.time()
# sentences = pickle.load(open('data/training_data', 'rb'))
# seqlen = []
# embedded_train_X = []
# embedded_train_Y = []
# onehot_train_X = []
# onehot_train_Y = []
# counter = 0
# postfix = -1
# for sentence in sentences:
#     if max_test_length >= len(sentence) >= MIN_SENTENCE_LENGTH:
#         valid_sentence_flag = 1
#         for word in sentence:
#             if word not in word_vector or word not in word_dict:
#                 valid_sentence_flag = 0
#                 break
#         if valid_sentence_flag == 1:
#             if len(sentence) <= seq_max_length:
#                 seqlen.append(len(sentence)-1)
#             else:
#                 seqlen.append(seq_max_length)
#             embedded_train_X.append([word_vector[sentence[i]] if i < len(sentence)-1 and sentence[i] in word_vector else np.zeros(WORD_VECTOR_SIZE) for i in range(seq_max_length)])
#             onehot_train_X.append([one_hot_dict[sentence[i]] if i < len(sentence)-1 and sentence[i] in word_dict else -1 for i in range(seq_max_length)])
#             if len(sentence) <= seq_max_length:
#                 embedded_train_Y.append(word_vector[sentence[-1]])
#                 # embedded_train_Y.append([word_vector[sentence[i]] if i < len(sentence) and sentence[i] in word_vector else np.zeros(WORD_VECTOR_SIZE) for i in range(1, MIN_SENTENCE_LENGTH)])
#                 onehot_train_Y.append(one_hot_dict[sentence[-1]])
#                 # onehot_train_Y.append([one_hot_dict[sentence[i]] if i < len(sentence) and sentence[i] in word_dict else -1 for i in range(1, MIN_SENTENCE_LENGTH)])
#             else:
#                 embedded_train_Y.append(word_vector[sentence[seq_max_length]])
#                 onehot_train_Y.append(one_hot_dict[sentence[seq_max_length]])

#             counter += 1
#             if counter % training_data_size == (training_data_size-1):
#                 postfix += 1
#                 print("Save file, postfix:", postfix)
#                 pickle.dump((np.array(embedded_train_X), np.array(embedded_train_Y), np.array(seqlen), len(embedded_train_X)), open('data/embedded_train_data_%d' % (postfix), 'wb'), True)
#                 pickle.dump((np.array(onehot_train_X), np.array(onehot_train_Y), np.array(seqlen), len(onehot_train_X)), open('data/onehot_train_data_%d' % (postfix), 'wb'), True)
#                 seqlen = []
#                 embedded_train_X = []
#                 embedded_train_Y = []
#                 onehot_train_X = []
#                 onehot_train_Y = []
# pickle.dump({
#         'postfix': postfix,
#         'seq_max_length': seq_max_length,
#         'max_test_length': max_test_length,
#         'min_test_length': min_test_length,
#         'max_space_index': max_space_index,
#         'min_space_index': min_space_index,
#         'onehot_vec_size': len(word_dict.keys()),
#         'wordvec_size': WORD_VECTOR_SIZE
#     }, open('data/embedded_data_config', 'wb'), True)
# print("len(seqlen)", len(seqlen))
# print("len(embedded_train_X)", len(embedded_train_X))
# print("len(embedded_train_Y)", len(embedded_train_Y))
# print("len(onehot_train_X)", len(onehot_train_X))
# print("len(onehot_train_Y)", len(onehot_train_Y))
# # pickle.dump((np.array(embedded_train_X), np.array(embedded_train_Y)), open('data/embedded_train_data', 'wb'), True)
# # pickle.dump((np.array(onehot_train_X), np.array(onehot_train_Y)), open('data/onehot_train_data', 'wb'), True)
# print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# load model
# word_dict = pickle.load(open('model/word_dict', 'rb'))
# one_hot_dict = pickle.load(open('model/one_hot_dict', 'rb'))
# word_vector = word2vec.Word2Vec.load_word2vec_format('model/word_vector.txt', binary=False)
# word_vector = word2vec.Word2Vec.load_word2vec_format('model/word_vector.bin', binary=True)
# (train_X, train_Y) = pickle.load(open('data/embedded_train_data', 'rb'))
# (train_X, train_Y) = pickle.load(open('data/onehot_train_data', 'rb'))
# print(word_vector['this'])
# print(word_vector['you'])
