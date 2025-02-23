# 根据用户 学习动力，能力提升意愿，兴趣度，空余时间，判断是否适合学习某个课程
# 逻辑回归模型使用的是 一个线性回归模型，将特征值带入到sigmoid函数中，得到的值大于0.5则为1，小于0.5则为0
# 决策树模型使用的是 一个树形结构，将特征值带入到树中，根据特征值的大小，逐步判断，最终得到结果


# 决策树的三种特征选择算法：ID3，C4.5，CART

from sklearn.datasets import load_iris
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 加载数据集
iris = load_iris()

# 转换为 Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

y = iris_df["target"]
X = iris_df.drop(columns=["target"], axis=1)
# print(X.head())

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

y_predict = knn.predict(X)
accuracy = accuracy_score(y, y_predict)
# 准确率为0.96
print("accuracy_score", accuracy)

X_norm = StandardScaler().fit_transform(X)


# X1_mean = X.iloc[:, 0].mean()
# x1_norm_mean = X_norm[:, 0].mean()
# x1_std = X.iloc[:, 0].std()
# x1_norm_std = X_norm[:, 0].std()
# # 均值近似0 标准差近1
# print(X1_mean, x1_norm_mean, x1_std, x1_norm_std)

# print(X.iloc[:, 0].head())
# fig1 = plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# plt.hist(X.iloc[:, 0], bins=100)
# plt.subplot(1, 2, 2)
# plt.hist(X_norm[:, 0], bins=100)
# plt.show()

pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_norm)

# var_ratio = pca.explained_variance_ratio_
# print("var_ratio", var_ratio)  #  [0.72962445 0.22850762 0.03668922 0.00517871]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
# fig3 = plt.figure(figsize=(20, 10))
# plt.subplot(1, 2, 1)
# setosa = plt.scatter(X_pca[:, 0][y == 0], X_pca[:, 1][y == 0])
# versicolor = plt.scatter(X_pca[:, 0][y == 1], X_pca[:, 1][y == 1])
# virginica = plt.scatter(X_pca[:, 0][y == 2], X_pca[:, 1][y == 2])
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.legend((setosa, versicolor, virginica), ("setosa", "versicolor", "virginica"))
# plt.show()


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_pca, y)

y_predict = knn.predict(X_pca)
accuracy = accuracy_score(y, y_predict)
# 0.946
print("pca accuracy_score", accuracy)


def pca1():
    # 先使用原维度数进行PCA处理
    pca = PCA(n_components=4)
    # 返回降维后的标注化数据
    X_reduced = pca.fit_transform(X)
    # 方差比例s
    # var_ratio = pca.explained_variance_ratio_
    # print("var_ratio", var_ratio)

    # 可视化方差比例:
    # plt.bar([1,2,3,4],var_ratio)
    # plt.title('variance ratio of each component')
    # plt.xticks([1,2,3,4],['PC1','PC2', 'PC3', 'PC4'])
    # plt.ylabel('var_ratio')
    # plt.show()

    # setosa = plt.scatter(X_reduced[:, 0][y == 0], X_reduced[:, 1][y == 0])
    # versicolor = plt.scatter(X_reduced[:, 0][y == 1], X_reduced[:, 1][y == 1])
    # virginica = plt.scatter(X_reduced[:, 0][y == 2], X_reduced[:, 1][y == 1])
    # plt.legend((setosa, versicolor, virginica), ("setosa", "versicolor", "virginica"))
    # plt.xlabel("PC1")
    # plt.ylabel("PC2")
    # plt.show()
