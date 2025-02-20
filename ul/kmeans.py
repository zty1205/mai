import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
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

# kmeans'
# 分成3类
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
centers = kmeans.cluster_centers_  # 查看聚类中心

print("centers", centers)
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
# plt.show()


y_test = kmeans.predict([[80, 60]])
plt.scatter([80], [60], c="red", s=100, alpha=0.5)
# plt.show()
# print(y_test)

y_predict = kmeans.predict(X)
# print(pd.Series(y_predict).value_counts(), pd.Series(y).value_counts())
# 输出结果 分类好像不对，因为是无监督学习，所以没有准确率，只做了分类
# 1    72
# 2    70
# 0    58
#
# 2    72
# 0    70
# 1    58

accuracy = accuracy_score(y, y_predict)
# 准确率为0
print("accuracy_score", accuracy)

# 分类矫正
y_correct = []
for i in y_predict:
    if i == 0:
        y_correct.append(1)
    elif i == 1:
        y_correct.append(2)
    else:
        y_correct.append(0)
print(pd.Series(y_correct).value_counts(), pd.Series(y).value_counts())
accuracy2 = accuracy_score(y, y_correct)  # 1.0
print("accuracy_score2", accuracy2)
