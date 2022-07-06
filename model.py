import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tflearn
import tensorflow
import json
import random
import pickle

with open('intents.json') as intents_file:
    data = json.load(intents_file)

stemmer = LancasterStemmer()

words = []
labels = []

docs_x = []
docs_y = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        tokenized_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_words)
        docs_x.append(tokenized_words)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

words = [stemmer.stem(word.lower()) for word in words if word not in "[$&+,:;=?@#|'<>.^*()%!]0123456789"]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

for x, doc in enumerate(docs_x):
    bag = []
    tokenized_words = [stemmer.stem(word) for word in doc]

    for w in words:
        if w in tokenized_words:
            bag.append(1)
        else:
            bag.append(0)

    output_row = [0 for _ in range(len(labels))]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open('data.pickle', 'wb') as data_file:
    pickle.dump((words, labels, training, output), data_file)

tensorflow.compat.v1.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation='softmax')
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')