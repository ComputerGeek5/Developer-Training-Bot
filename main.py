import discord
import os
from dotenv import load_dotenv
import tflearn
import tensorflow
import pickle
import json
import random
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy
from os import environ
from flask import Flask

# Loading trained data
with open('intents.json') as intents_file:
    data = json.load(intents_file)

with open('data.pickle', 'rb') as data_file:
    words, labels, training, output = pickle.load(data_file)

tensorflow.compat.v1.reset_default_graph()

network = tflearn.input_data(shape=[None, len(training[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(output[0]), activation='softmax')
network = tflearn.regression(network)

model = tflearn.DNN(network)

model.load('model.tflearn')

# Bot response
def bot_response(user_input):
    results = model.predict([bag_of_words(user_input, words)])[0]
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    if results[results_index] > 0.7:
        for intent in data['intents']:
            if tag == intent['tag']:
                responses = intent['responses']

        return random.choice(responses)
    else:
        return 'I did not understand. Can you please try again?'

def bag_of_words(user_input, words):
    bag = [0 for _ in range(len(words))]
    tokenized_words = nltk.word_tokenize(user_input)

    stemmer = LancasterStemmer()
    tokenized_words = [stemmer.stem(word.lower()) for word in tokenized_words]

    for tokenized_word in tokenized_words:
        for index, word in enumerate(words):
            if word == tokenized_word:
                bag[index] = 1

    return numpy.array(bag)


bot = discord.Client()

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    if isinstance(message.channel, discord.channel.DMChannel) and message.author != bot.user:
        bot_message = bot_response(message.content)
        await message.channel.send(bot_message)

with open('token.txt') as token_file:
    TOKEN = token_file.readline()

bot.run(TOKEN)
