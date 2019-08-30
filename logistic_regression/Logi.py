# -*- coding: utf-8 -*-


###Task 2 使用sklearn逻辑回归进行分类
##输入数据为vehicle.csv
#要求：
#（1）针对模型构建分类模型
#（2）画出分类边界
#提示：model.coef_ & model.intercept_


########  你的可视化结果看起来应该类似于'demo'  ##############

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
#import sklearn
#print(sklearn.__version__)

model = LogisticRegression(
                            C=1.0, 
                            class_weight=None, 
                            multi_class='ovr', 
                            penalty='l2', 
                            solver='liblinear'
                           )

data = pd.read_csv('vehicle.csv')

labels = data['label']
feature = data.drop(['label'],axis = 1)

from sklearn.model_selection import train_test_split
feature_train, feature_test, label_train, label_test = train_test_split(feature, labels,test_size = 0.2 ,random_state = 1001)

model.fit(feature_train, label_train)

print('model.coef_ =', model.coef_)
print('model.intercept_ =', model.intercept_)

coef = model.coef_[0]
intercept = model.intercept_
x_plot = np.arange(5.9, 6.4, 0.01)
y_plot = (-x_plot*coef[0]-intercept)/coef[1]

plt.plot()
plt.scatter(data['length'][data['label'] == 'car'], data['width'][data['label'] == 'car'], c='#66ffff')
plt.scatter(data['length'][data['label']=='truck'], data['width'][data['label']=='truck'], c='r')
plt.plot(x_plot,y_plot,'g-')
plt.show()