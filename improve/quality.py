import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("improve/quality.csv")

X = data.drop(columns=["y"], axis=1)
y = data.loc[:, "y"]

X1 = X.loc[:, "x1"]
X2 = X.loc[:, "x2"]

fig1 = plt.figure(figsize=(8, 6))


def show():
    bad = plt.scatter(X1[y == 0], X2[y == 0], color="blue")
    good = plt.scatter(X1[y == 1], X2[y == 1], color="red")
    plt.legend((bad, good), ("bad", "good"))
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()


## 根据高斯分布剔除异常点


def gs():
    ad_model = EllipticEnvelope(contamination=0.02)
    ad_model.fit(X[y == 0])
    y_predict_bad = ad_model.predict(X[y == 0])
    # print(y_predict_bad)  # -1就是异常点

    plt.scatter(
        X1[y == 0][y_predict_bad == -1],
        X2[y == 0][y_predict_bad == -1],
        marker="x",
        s=150,
    )
    plt.show()


# 手动清楚异常点 6.9,9.9,0


# PCA降维


def pca():
    X = StandardScaler().fit_transform(X)

    X_norm = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_norm)
    # 降维后的各主成分的方差值。方差值越大,则说明越是重要的主成分
    var_radio = pca.explained_variance_ratio_
    print("var_radio: ", var_radio)  # [0.51688119 0.48311881]


# 数据分离
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# print(
#     X_train.shape,
#     X_test.shape,
#     X_reduced.shape,
# )

# 建立knn模型


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

# accuracy_train = accuracy_score(y_train, y_train_predict)
# accuracy_test = accuracy_score(y_test, y_test_predict)

# print("accuracy_train: ", accuracy_train)  # 0.85
# print("accuracy_test: ", accuracy_test)  # 0.58


# 可视化边界
def area():
    xx, yy = np.meshgrid(np.arange(0, 10, 0.05), np.arange(0, 10, 0.05))
    x_range = np.c_[xx.ravel(), yy.ravel()]
    # print(x_range)
    y_range_predict = knn.predict(x_range)

    knn_bad = plt.scatter(
        x_range[:, 0][y_range_predict == 0],
        x_range[:, 1][y_range_predict == 0],
        color="blue",
    )
    knn_good = plt.scatter(
        x_range[:, 0][y_range_predict == 1],
        x_range[:, 1][y_range_predict == 1],
        color="red",
    )
    bad = plt.scatter(X1[y == 0], X2[y == 0], color="blue")
    good = plt.scatter(X1[y == 1], X2[y == 1], color="red")
    plt.legend((bad, good, knn_bad, knn_good), ("bad", "good", "knn_bad", "knn_good"))
    plt.show()


# 混淆矩阵


from sklearn.metrics import confusion_matrix


def quality():

    cm = confusion_matrix(y_test, y_test_predict)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    print("TP: ", TP)  # 6
    print("TN: ", TN)  # 3
    print("FP: ", FP)  # 3
    print("FN: ", FN)  # 4

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    print("Accuracy: ", Accuracy)  # 0.5625
    print("Precision: ", Precision)  # 0.66666
    print("Recall: ", Recall)  # 0.6
    print("F1: ", F1)  # 0.6135


# 尝试不同的 neighbors 数量
def neighbors():
    n = np.arange(1, 21)
    train_accuracy = np.empty(len(n))
    test_accuracy = np.empty(len(n))

    for i, k in enumerate(n):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    plt.plot(n, train_accuracy, label="train_accuracy", marker="o")
    plt.plot(n, test_accuracy, label="test_accuracy", marker="o")
    plt.xlabel("neighbors")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    # 可视化后发现，当 neighbors=1 时，模型效果最好


neighbors()
