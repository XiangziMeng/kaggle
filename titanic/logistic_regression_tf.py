# coding: utf-8

import sys 
sys.path.insert(0, "/home/pi/repository/machine_learning_algorithms")
import math
import numpy as np
import pandas as pd

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.config.experimental import list_physical_devices, set_memory_growth
physical_devices = list_physical_devices('GPU')
set_memory_growth(physical_devices[0], True)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model





train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

features_1 = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
features_2 = ["Sex", "Embarked"]


def bucket(num_list, low, high, size):
    dim = math.ceil((high - low) / size) + 1
    vector_list = []
    res = []
    for num in num_list:
        num = float(num)
        if math.isnan(num):
            num = high
        elif num < low or num > high:
            num = high
        vector = [0] * dim
        for k in range(math.floor((num - low) / size)+1):
            vector[k] = 1
        vector_list.append(vector)
    for i in range(dim):
        features = []
        for j in range(len(vector_list)):
            features.append(vector_list[j][i])
        res.append(features)
    return res
    
def category(f_name, feature_list):
    dim = 0
    vector_list = []
    res = []
    for feature in feature_list:
        if f_name == "Sex":
            dim = 2
            if feature == "male":
                vector = [1, 0]
            if feature == "female":
                vector = [0, 1]
        if f_name == "Embarked":
            dim = 3
            if feature == "S":
                vector = [1, 0, 0]
            if feature == "C":
                vector = [0, 1, 0]
            if feature == "Q":
                vector = [0, 0, 1]
        vector_list.append(vector)
    for i in range(dim):
        features = []
        for j in range(len(vector_list)):
            features.append(vector_list[j][i])
        res.append(features)
    return res


def transform(data):
    dic = {}
    for f_name in features_1:
        num_list = np.array(data[f_name])
        if f_name == "Pclass":
            low, high, size = 1, 3, 1
        elif f_name == "Age":
            low, high, size = 0, 120, 5
        elif f_name == "SibSp":
            low, high, size = 0, 8, 1
        elif f_name == "Parch":
            low, high, size = 0, 6, 1
        elif f_name == "Fare":
            low, high, size = 0, 160, 5
        else:
            continue
        tmp = bucket(num_list, low, high, size)
        dim = len(tmp)
        for i in range(dim):
            dic[f_name + "_" + str(i)] = tmp[i]
    for f_name in features_2:
        feature_list = np.array(data[f_name])
        tmp = category(f_name, feature_list)
        dim = len(tmp)
        for i in range(dim):
            dic[f_name + "_" + str(i)] = tmp[i]        
    return pd.DataFrame(dic)

def transform_v2(data):
    res = []
    for f_name in features_1:
        num_list = np.array(data[f_name])
        if f_name == "Pclass":
            low, high, size = 1, 3, 1
        elif f_name == "Age":
            low, high, size = 0, 120, 5
        elif f_name == "SibSp":
            low, high, size = 0, 8, 1
        elif f_name == "Parch":
            low, high, size = 0, 6, 1
        elif f_name == "Fare":
            low, high, size = 0, 160, 5
        else:
            continue
        tmp = bucket(num_list, low, high, size)
        res.extend(tmp)
    for f_name in features_2:
        feature_list = np.array(data[f_name])
        tmp = category(f_name, feature_list)
        res.extend(tmp)
    # add feature 常值1
    tmp = [1] * len(num_list)
    res.extend([tmp])
    return np.transpose(np.array(res, dtype=np.float32))


X_train = transform_v2(train_data)
X_test = transform_v2(test_data)

y_train = np.array(train_data["Survived"], dtype=np.float32)


print(X_train.shape, X_test.shape, y_train.shape)

n_samples = X_train.shape[0]
n_features = X_train.shape[1]

class MyModel(Model):
  def __init__(self, n_features):
    super(MyModel, self).__init__()
    self.w = tf.Variable([0.1] * n_features, trainable=True)
    self.b = tf.Variable(0.5, trainable=True)

  def call(self, x):
    r = tf.reduce_sum(tf.multiply(self.w, x),axis=1)
    r = tf.add(r, self.b)
    #r = tf.math.sigmoid(r)
    r = -tf.multiply(r, r)
    r = 1 - tf.math.exp(r)
    return r

  def predict(self, x):
    preds = self.call(x)
    res = []
    for pred in preds:
        if pred > 0.5:
            res.append(1)
        else:
            res.append(0)
    print(res)
    return res

  def product(self, x):
    r = tf.reduce_sum(tf.multiply(self.w, x),axis=1)
    r = tf.add(r, self.b)
    return r

model = MyModel(n_features)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def train_step(X_train, y_train):
  with tf.GradientTape() as tape:
    y_pred = model(X_train, training=True)
    product = model.product(X_train)
    #loss = -tf.reduce_sum(tf.multiply(y_train, y_pred)) + tf.reduce_sum(tf.math.log(tf.math.exp(y_pred)+1))
    loss = tf.reduce_sum(tf.multiply(product, product)) - tf.reduce_sum(tf.multiply(y_train, tf.math.log(tf.math.exp(tf.multiply(product, product)) - 1)))
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def training_acc(y, y_pred, n_samples):
    res = 0
    for i in range(n_samples):
        if int(y[i]) == 1 and y_pred[i] > 0.5:
            res += 1
        elif int(y[i]) == 0 and y_pred[i] < 0.5:
            res += 1
    res = res * 1.0 / n_samples
    return res

EPOCHS = 1000
for epoch in range(EPOCHS):
    train_step(X_train, y_train)
    y_pred = model(X_train, training=True)
    product = model.product(X_train)
    #loss = -tf.reduce_sum(tf.multiply(y_train, y_pred)) + tf.reduce_sum(tf.math.log(tf.math.exp(y_pred)+1))
    loss = tf.reduce_sum(tf.multiply(product, product)) - tf.reduce_sum(tf.multiply(y_train, tf.math.log(tf.math.exp(tf.multiply(product, product)) - 1)))
    acc = training_acc(y_train, y_pred, n_samples)
    print(loss, acc)

predictions = model.predict(X_test)

print(predictions[:20])

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('data/submission_tf_2.csv', index=False)
print("Your submission was successfully saved!")
    
