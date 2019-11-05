import numpy as np
import tensorflow as tf


def create_lookup_table(data):
    # 单词去重
    words = list(set(data))
    vocab_to_index = {word: idx for idx, word in enumerate(words)}
    index_to_vocab = {vocab_to_index[word]: word for word in words}
    return vocab_to_index, index_to_vocab


data = ['long ago , the mice had a general council to consider',
        'long ago , the mice had a general council to consider',
        'long ago , the mice had a general council to consider']

text = []
for line in data:
    line = line.strip().lower()
    for word in line.split(" "):
        text.append(word)

vocab_to_index, index_to_vocab = create_lookup_table(text)

X = []
Y = []
number_time_steps = 3
for content in data:
    words = content.strip().split(" ")
    words_number = len(words)
    offset = 0
    while offset < words_number - number_time_steps:
        tmp_x = words[offset:offset + number_time_steps]  # List[0:3] [1:4]
        tmp_y = words[offset + number_time_steps]  # List[3] [4]
        X.append([vocab_to_index[tx] for tx in tmp_x])
        Y.append(vocab_to_index[tmp_y])
        offset += 1
        # print("{}---->{}".format(tmp_x, tmp_y))

X = np.asarray(X).reshape((-1, number_time_steps))
Y = np.asarray(Y).reshape(-1)

total_samples = np.shape(X)[0]
total_batch = total_samples // 8
print(X)
print(total_samples)
print(total_batch)
random_index = np.random.permutation(total_samples)
print(random_index)
idx = random_index[8:16]
train_x = X[idx]
print(train_x)
