import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("ul/data.csv")
# print(data.head())

X = data.drop(columns=["labels"], axis=1)
y = data.loc[:, "labels"]
# 统计
# print(pd.Series(y).value_counts())

# 画图
fig1 = plt.figure()
label0 = plt.scatter(X.loc[:, "V1"][y == 0], X.loc[:, "V2"][y == 0], color="red")
label1 = plt.scatter(X.loc[:, "V1"][y == 1], X.loc[:, "V2"][y == 1], color="blue")
label2 = plt.scatter(X.loc[:, "V1"][y == 2], X.loc[:, "V2"][y == 2], color="green")
plt.xlabel("V1")
plt.ylabel("V2")
plt.legend((label0, label1, label2), ("label0", "label1", "label2"))
# plt.show()

# 预先给出分类
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)


y_test = knn.predict([[80, 60]])
plt.scatter([80], [60], c="red", s=100, alpha=0.5)
plt.show()
print(y_test)

y_predict = knn.predict(X)


accuracy = accuracy_score(y, y_predict)
# 准确率为1
print("accuracy_score", accuracy)

