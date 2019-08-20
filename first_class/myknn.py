# print(1)
# print(0b11)
# print(0xff)

# b = tuple([1, 2, 3, [1, 2, 3]])
# b[3][1] = 100
# print(b)

# import numpy as np
# def cal_dist_L2(A, B):
#     distance_L2=np.sqrt(np.sum((A-B)**2))
#     return distance_L2
#
# A=np.array([4.8, 2.2])
# B=np.array([4.7, 2.1])
#
# distance=cal_dist_L2(A, B)
# print(distance)

# import libs
import numpy as np
import pandas as pd

import matplotlib.pylab as plt

data = pd.read_csv('vehicle.csv')

feature = np.array(data.iloc[:, 0:2])
labels = data['label'].tolist()

plt.scatter(data['length'][data['label'] == 'car'], data['width'][data['label'] == 'car'], c='y')
plt.scatter(data['length'][data['label'] == 'truck'], data['width'][data['label'] == 'truck'], c='r')

test = [4.7, 2.1]
### step 1, calculate distance ###

numSamples = data.shape[0]  # 读取行数

# np.tile() 函数，就是将原矩阵横向、纵向地复制
# 这里复制了成和我们数据一样多的行数，再进行相减
diff = np.tile(test, (numSamples, 1)) - feature

# diff 里的每项都平方
squareDiff = diff ** 2

# 横向累加
squareDist = np.sum(squareDiff, axis=1)

distance = squareDist ** 0.5

### step 2, sort the distance ###

# 拿到排序后的索引
sortDistIndices = np.argsort(distance)

### step 3, find the K nearest points ###

k = 9

classCount = {}
label_count = []

for i in range(k):
    voteLabel = labels[sortDistIndices[i]]
    classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    label_count.append(voteLabel)

### step 4, count the numbers and get result ###

from collections import Counter

word_counts = Counter(label_count)
top = word_counts.most_common(1)
