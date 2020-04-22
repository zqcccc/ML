
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt

data = pd.read_csv('vehicle.csv')

target = [6.4, 2.9]
target = np.array(target).reshape(1, -1)

# 定义均值，返回为均值矩阵（矩阵大小为类别数*特征数，本例中是2*2）
def get_mean(data):
    return data.groupby('label').mean()

# 定义标准差，返回标准差矩阵
def get_std(data):
    return data.groupby('label').std()

# 定义正态分布概率密度，返回为矩阵（矩阵大小为类别数*特征数，本例中是2*2）
def gaussian_probability(target, data):
    mean = get_mean(data)
    std = get_std(data)
    return np.exp((-(target-mean)**2)/(2*std*std))/(np.sqrt(2*np.pi)*std)

# 依据数据的类别，正分布概率密度，计算每个类别对应的概率（返回大小为向量，长度为类别数）
def class_probability(gaussian_proba, data):
    p_joint = 1
    count_z = data.groupby('label').count()
    p_z = count_z / data['label'].count()
    for i in range(len(gaussian_proba.iloc[0])):
        p_joint = p_joint * gaussian_proba.iloc[:,i]
    p_class = p_z.iloc[:,0] * p_joint

    return p_class

# 依据返回的类别概率，拿概率最高的类别作为结果
def get_class(p_class):
    p_class = p_class.sort_values(ascending = False)
    return p_class.index[0]

gaussian_proba = gaussian_probability(target, data)
class_proba = class_probability(gaussian_proba, data)

print('classificatior result is ' + get_class(class_proba))