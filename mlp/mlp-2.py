# 读取csv文件， 数据格式为x1, x2, y，并画出散点图 y=0为红色，y=1为蓝色
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("mlp-2.data.csv")


def char():
  data0 = data[data["y"] == 0]
  data1 = data[data["y"] == 1]
  plt.scatter(data0["x1"], data0["x2"], color="blue")
  plt.scatter(data1["x1"], data1["x2"], color="orange")
  plt.show()



X = data.drop("y", axis=1)
y = data.loc[:, "y"]
# 数据分离
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)
 
# print(X_train.shape, X_test.shape, X.shape) 


from keras.models import Sequential
from keras.layers import Dense, Activation


mlp = Sequential()
# 20个神经元的隐藏层，输入层有2个神经元，激活函数为sigmoid
mlp.add(Dense(units=20, input_dim = 2, activation='sigmoid'))
# 输出层有1个神经元，激活函数为sigmoid
mlp.add(Dense(units=1, activation='sigmoid'))
# 打印模型结构
# mlp.summary()


'''
Model: "sequential"  不会包含输入层
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 20)                60        
                                                                 
 dense_1 (Dense)             (None, 1)                 21        
                                                                 
=================================================================
Total params: 81 (324.00 Byte)
Trainable params: 81 (324.00 Byte)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

'''

# # 编译模型 使用adam优化器，损失函数为binary_crossentropy，评估标准为accuracy
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# # 训练模型，训练1000次，每次训练10个数据
mlp.fit(X_train, y_train, epochs=1000)
# # 评估模型
y_pred_probs = mlp.predict(X_train) # n行1列的数组，每个元素是一个概率 0-1
# print('y_pred_probs', y_pred_probs)
# 将概率转换为类别
y_train_predict = (y_pred_probs > 0.5).astype("int32") # predict_classes已经被弃用

from sklearn.metrics import accuracy_score
# 计算准确率
# accuracy_train = accuracy_score(y_train, y_train_predict)
# print("Train accuracy: ", accuracy_train)

# 可视化模型预测结果
print(y_train_predict)  # n行1列的数组

y_train_format = pd.Series(i[0] for i in y_train_predict)
print(y_train_format) # 一维数组

# # 将预测结果转换为可用于索引的Series
# y_range_predict = pd.Series(i[0] for i in y_range_predict)

import numpy as np
xx, yy = np.meshgrid(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01))
x_range = np.c_[xx.ravel(), yy.ravel()]
y_range_probs = mlp.predict(x_range)
y_range_predict = (y_range_probs > 0.5).astype("int32")
y_range_predict = pd.Series(i[0] for i in y_range_predict)

def char_model():
  data0 = data[data["y"] == 0]
  data1 = data[data["y"] == 1]
  
  passed_predict = plt.scatter(x_range[:,0][y_range_predict == 1], x_range[:,1][y_range_predict == 1], color="red")
  failed_predict = plt.scatter(x_range[:,0][y_range_predict == 0], x_range[:,1][y_range_predict == 0], color="green")
  
  passed = plt.scatter(data1["x1"], data1["x2"], color="orange")
  failed = plt.scatter(data0["x1"], data0["x2"], color="blue")
  
  plt.legend((passed, failed, passed_predict, failed_predict), ("passed", "failed", "passed_predict", "failed_predict"))
  plt.xlabel("x1")
  plt.ylabel("x2")
  plt.show()

# char_model()