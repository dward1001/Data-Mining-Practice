
# coding: utf-8

# In[ ]:

import tensorflow as tf
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

train_data = pd.read_csv('train.csv')
dfTrainFeatureVectors = train_data.drop(['label'], axis=1)
trainFeatureVectors = dfTrainFeatureVectors.values.astype(dtype=np.float32)
trainFeatureVectorsConvoFormat = trainFeatureVectors.reshape(42000, 28, 28, 1)

trainLabelsList = train_data['label'].tolist()
trainLabelsTensor = tf.one_hot(trainLabelsList, depth=10)
trainLabelsNdarray = tf.Session().run(trainLabelsTensor).astype(dtype=np.float64)

test_data = pd.read_csv('test.csv')
testFeatureVectors = test_data.values.astype(dtype=np.float32)
testFeatureVectorsConvoFormat = testFeatureVectors.reshape(28000, 28, 28, 1)


X = tf.placeholder('float32', shape = [None, 28, 28, 1])
y = tf.placeholder('float32', shape = [None, 10])
lr = tf.placeholder('float32')
keep_prob = tf.placeholder('float32', name = "keep_prob")

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, 6], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, 'float32', [6]))
conv1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

W2 = tf.Variable(tf.truncated_normal([5, 5, 6, 12], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, 'float32', [12]))
conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)

W3 = tf.Variable(tf.truncated_normal([4, 4, 12, 24], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, 'float32', [24]))
conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W3, strides=[1, 2, 2, 1], padding='SAME') + b3)
conv3_ = tf.reshape(conv3, shape=[-1, 7 * 7 * 24])

W4 = tf.Variable(tf.truncated_normal([7 * 7 * 24, 200], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, 'float32', [200]))
fc1 = tf.nn.relu(tf.matmul(conv3_, W4) + b4)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

W5 = tf.Variable(tf.truncated_normal([200, 10], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, 'float32', [10]))
fc2 = tf.matmul(fc1_drop, W5) + b5
y_ = tf.nn.softmax(fc2)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc2, labels=y)
cross_entropy = tf.reduce_mean(cross_entropy)*100

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predictions = tf.argmax(y_, 1)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

def get_batch(i, size, tr_features, tr_labels):
    start = (i * size) % 32000
    batch_X = tr_features[start : start+size]
    batch_Y = tr_labels[start : start+size]

    return batch_X, batch_Y

for i in range(10000+1): 
    batch_X, batch_Y = get_batch(i, 100, trainFeatureVectorsConvoFormat, trainLabelsNdarray)

    max_rate = 0.003
    min_rate = 0.0001
    decay = 2000.0
    learning_rate = min_rate + (max_rate - min_rate) * math.exp(-i/decay)

    if i % 20 == 0:
        sess.run([accuracy, cross_entropy], {X: batch_X, y: batch_Y, keep_prob: 1.0})

    if i % 100 == 0:
        sess.run([accuracy, cross_entropy], {X: trainFeatureVectorsConvoFormat[-10000:], y: trainLabelsNdarray[-10000:], keep_prob: 1.0})
    
    sess.run(train_step, {X: batch_X, y: batch_Y, lr: learning_rate, keep_prob: 0.75})

sess.run([accuracy, cross_entropy], {X: trainFeatureVectorsConvoFormat[-10000:], y: trainLabelsNdarray[-10000:], keep_prob: 1.0})
p = sess.run([predictions], {X: testFeatureVectorsConvoFormat, keep_prob: 1.0})

results = pd.DataFrame({'ImageId': pd.Series(range(1, len(p[0]) + 1)), 'Label': pd.Series(p[0])})
results.to_csv('mnist_result.csv', index=False)

