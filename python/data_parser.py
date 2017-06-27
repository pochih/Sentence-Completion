from __future__ import print_function
from os import listdir
import re
import time
import codecs
import pickle


""" Extract only the vocabulary part of the data """
def refine(data):
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words]
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words)
    return data


# parse testing data
print("Parsing testing data...")
ts = time.time()
testing_data = []
with codecs.open('model/testing_data_words.txt', "w", encoding='utf-8', errors='ignore') as out:
    with open('data/testing_data.csv', 'r') as f:
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
            out.write(line + ' ')
pickle.dump(testing_data, open('data/testing_data', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# parse training data
print("Parsing training data...")
ts = time.time()
training_data = []
with codecs.open('model/all_words.txt', "w", encoding='utf-8', errors='ignore') as out:
    for filename in listdir('data/Holmes_Training_Data'):
        # with open('data/Holmes_Training_Data/%s' % filename, 'r') as f:
        with codecs.open('data/Holmes_Training_Data/%s' % filename, "r", encoding='utf-8', errors='ignore') as f:
            parseFlag = False
            data = ""
            for line in f:
                if line.find('*END*THE SMALL PRINT!') > -1 \
                    or line.find('ENDTHE SMALL PRINT!') > -1:  # remove some useless paragraphs at the beginning
                    parseFlag = True
                    continue
                if parseFlag:
                    line = line.replace("\r\n", " ")
                    data += line.lower()
            sentences = data.split('.')
            training_data += [refine(sentence).split() for sentence in sentences]
            data = refine(data)
            out.write(data + ' ')
pickle.dump(training_data, open('data/training_data', 'wb'), True)
print("Time Elapsed: {0} secs\n".format(time.time() - ts))


# load data
# testing_data = pickle.load(open('data/testing_data', 'rb'))
# training_data = pickle.load(open('data/training_data', 'rb'))
