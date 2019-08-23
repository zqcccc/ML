# -*- coding: utf-8 -*-


###Task 2 使用sklearn解决回归问题
##输入数据为linear_data.csv
#要求：
#（1）针对模型构建回归模型
#（2）可视化出回归直线
#（3）*使用你自己定义函数的计算出MSE，不允许使用sklearn的API
#（4）*** 选做，不计入考评分数，使用poly_data.csv，分别构建线性回归和多项式回归模型，计算RMSE，并进行可视化
# 提示：相关API：sklearn.preprocessing.PolynomialFeatures

########  这里的可视化结果看起来应该类似于'demo.png'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
# * from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 这里写成绝对路径的原因是相对路径有时候用不了，很诡异
# data = pd.read_csv('E:/ML/机器学习/线性回归/练习/linear_data.csv')
data = pd.read_csv('linear_data.csv')

labels = data['hx']
data = data['x'].values.reshape(-1, 1)

plt.scatter(data,labels,c='black')

model_line = LinearRegression(normalize = True)

model_line.fit(data,labels)

x_plot = np.arange(0, 11, 0.01)
y_plot = model_line.predict(x_plot.reshape(-1, 1))

plt.scatter(data,labels,c='black')
plt.plot(x_plot,y_plot,'r-')

#import metrics_ext

def metrics_MAE(x1,x2):
    
    #你的定义
    x1 = np.array(x1)
    x2 = np.array(x2)
    score = 0
    for i in range(x1.size):
        score = score + abs(x1[i]-x2[i])
    
    score = score/x1.size
    return score

print('MAE' , metrics_MAE(labels, model_line.predict(data.reshape(-1,1))))


########  下面我的可视化结果看起来应该类似于'demo/demo_poly.png'  ##############

poly_data = pd.read_csv('poly_data.csv')

poly_labels = poly_data['hx']
poly_data = poly_data['x'].values.reshape(-1, 1)

plt.scatter(poly_data, poly_labels,c='black')

model_line_ploy = LinearRegression(normalize = True)

model_line_ploy.fit(poly_data, poly_labels)

poly_x_plot = np.arange(0, 8.1, 0.01)
poly_y_plot = model_line_ploy.predict(poly_x_plot.reshape(-1, 1))

linner_r_score = model_line_ploy.score(poly_data,poly_labels)
print('r square', linner_r_score)

from sklearn.preprocessing import PolynomialFeatures

# 其中degree就是我们要处理的自变量的指数，如果degree = 1，就是普通的线性回归。
model_multiple_ploy = PolynomialFeatures(degree=2)

# 在处理多项式回归的过程中，需要使用fit_transform函数对训练集数据先进行拟合，然后再标准化，然后对测试集数据使用transform进行标准化，属于数据预处理的一种方法。
# 对训练集进行拟合标准化处理
x_transformed = model_multiple_ploy.fit_transform(poly_data)

# 模型初始化
poly_linear_model = LinearRegression(normalize = True)

# 拟合
poly_linear_model.fit(x_transformed, poly_labels)
# 预测
poly_prediction = poly_linear_model.predict(model_multiple_ploy.transform(poly_x_plot.reshape(-1, 1)))

plt.scatter(poly_data, poly_labels,c='black')
plt.plot(poly_x_plot, poly_y_plot,'r-', label='linear')
plt.plot(poly_x_plot, poly_prediction,'b-', label='poly')
plt.xlabel('x')
plt.ylabel('hx',rotation=0)
plt.title('regression')
plt.legend(loc='upper right')

poly_r_score = poly_linear_model.score(x_transformed, poly_labels)
print('r square', poly_r_score)


# RMSE
def metrics_RMSE(x1,x2):
    
    #你的定义
    x1 = np.array(x1)
    x2 = np.array(x2)
    score = 0
    for i in range(x1.size):
        score = score + pow(x1[i]-x2[i], 2)
    
    score = np.sqrt(score/x1.size)
    
    return score

o_linear_prediction = model_line_ploy.predict(poly_data)
o_poly_prediction = poly_linear_model.predict(x_transformed)

print('linear RMSE', metrics_RMSE(poly_labels, o_linear_prediction))
print('poly RMSE', metrics_RMSE(poly_labels, o_poly_prediction))
# poly 的误差明显更小， r squared 更接近 1
