import os
import numpy as np
import tensorflow as tf

path = "belling_the_cat.txt"


def load_data(file_path):
    with open(file_path, 'r') as reader:
        data = reader.read()
    return data


data = load_data(path)


def create_lookup_table(data):
    words = list(set(data))
    vocab_to_index = {word: idx for idx, word in enumerate(words)}
    index_to_vocab = {vocab_to_index[word]: word for word in words}
    return vocab_to_index, index_to_vocab


vocab_to_index, index_to_vocab = create_lookup_table(data.split(" "))
print(vocab_to_index)
print(index_to_vocab)
