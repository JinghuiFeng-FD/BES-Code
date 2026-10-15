import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

import matplotlib
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
warnings.filterwarnings("ignore")
# matplotlib.use('TkAgg')


data1 = pd.read_csv('../CSVDataset/automobile.csv')
# data1 = pd.read_excel('D:\\论文\\论文数据\\实验处理得到的数据\\ASD\\特征选择个数\\50_0_117833_new.xlsx')
columns_names = data1.columns[:-1].values
# print("columns_names:", columns_names, len(columns_names))
data = np.array(data1.iloc[:, :])
data = pd.DataFrame(data)  # 从第一行第一列开始
X = data.iloc[:, :-1]  # 数据部分
# print("-----------------------------数据值-----------------------------------",)
# print(data.iloc[:, :-1])  # 目标值
# print("----------------------------+目标值+------------------------------------")
y = data.iloc[:, -1:].values.flatten()
# print(y)
chrom_length = len(data.columns) - 1
estimator = KNeighborsClassifier()
# estimator = svm.SVC(kernel='rbf', C=10, gamma=1)

def calculate_fitness(swarm,weight):
    # print("swarm:",swarm)
    X_test = X
    count = 0
    has_1 = False
    # drop_columns = np.where(swarm < 0)[0]
    # # print(drop_columns)
    # count = len(swarm)-len(drop_columns)
    # X_test = X_test.drop(columns=drop_columns)
    # if count != 0:
    #     has_1 = True

    for j in range(chrom_length):
        # print(swarm[j],swarm_up[j])
        # step = swarm[j]-swarm_up[j]
        # print(j)
        # if random.random() <= continuous2binary(swarm[j]):
        # if 0.5 <= continuous2binary(swarm[j]):
        if swarm[j] < 0:
            choice = 0
        else:
            choice = 1
        # print(j, choice)
        if choice:  # 如果该特征没有选上，则将该特征剔除
            X_test = X_test.drop(columns=j)
        else:
            has_1 = True
            count += 1  # 染色体中1即选中的特征的个数
    X_test = X_test.values
    # print(X_test.shape, count)

    if has_1:
        # print("--")
        x_train, x_test, y_train, y_test = train_test_split(X_test, y, random_state=2)
        # print("&&")
        estimator.fit(x_train,y_train)
        # print("$$")
        estimator_fitness = estimator.score(x_test,y_test)
        # print("**")
        # estimator_fitness = cross_val_score(estimator, X_test, y, cv=5).mean()  # 5次交叉验证
        # print(estimator_fitness,"验证1", cross_val_score(estimator, X_test, y, cv=5).mean())  # 5次交叉验证)
    else:
        estimator_fitness = 0

    return np.array([weight*(count/chrom_length)+(1-weight)*(1-estimator_fitness), count, 1-estimator_fitness])


def continuous2binary(x):
    return 1/(1+np.exp(-x))

def continuous2binary_1(x):
    return abs(x/(math.sqrt(x**2+1)))

# x = np.arange(-6,6,1)
# y = continuous2binary(x)
# plt.plot(x,y)
# plt.show()

# swarm = np.array([ 1.81437573e+00,  2.28384407e+00, -2.55041318e+00, -6.63241264e-01,
#  -1.68694659e-01, -1.60627418e+00, -3.87320418e+00, -2.02613806e+00,
#   2.29604134e+00,  8.59750098e-01, -4.66980098e+00, -2.39794323e+00,
#  -8.37008120e-01,  1.83825352e-01, -2.97227953e-02, -4.39177859e+00,
#  -6.15492914e-01, -1.45932393e+00, -9.33042377e-01, -1.84945437e+00,
#  -1.86195098e+00,  5.62023657e-04,  3.75692413e-01, -7.06110831e-01,
#   4.46480233e-01])
#
# calculate_fitness(swarm, 0.2)
