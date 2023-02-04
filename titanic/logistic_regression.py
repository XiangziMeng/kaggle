# coding: utf-8

import sys 
sys.path.insert(0, "/home/pi/repository/machine_learning_algorithms")
import math
import numpy as np
import pandas as pd
#from mlfromscratch.supervised_learning.logistic_regression import LogisticRegression
#from sklearn.linear_model import LogisticRegression
from logistic_regression_v2 import LogisticRegression

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
    return np.transpose(np.array(res))


X_train = transform_v2(train_data)
X_test = transform_v2(test_data)

y_train = np.array(train_data["Survived"])

model = LogisticRegression()

model.fit(X_train, y_train, n_iterations=200)

predictions = model.predict(X_test)

print(predictions[:20])
model.print_parameters()

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('data/submission.csv', index=False)
print("Your submission was successfully saved!")
