import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data = pd.read_csv("improve/enzyme.csv")

# 将data分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    data.loc[:, "Temperature"], data.loc[:, "Activity"], random_state=4, test_size=0.4
)

print(len(X_train), len(X_test))

X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

fig1 = plt.figure(figsize=(8, 6))
train = plt.scatter(X_train, y_train, color="red")
plt.title("Temperature vs Activity")
plt.xlabel("Temperature")
plt.ylabel("Activity")
# plt.show()


def lm():
    lm1 = LinearRegression()
    lm1.fit(X_train, y_train)

    y_train_predict = lm1.predict(X_train)
    y_test_predict = lm1.predict(X_test)

    r2_score_train = r2_score(y_train, y_train_predict)
    r2_score_test = r2_score(y_test, y_test_predict)

    print("r2_score_train:", r2_score_train)  # 0.336
    print("r2_score_test:", r2_score_test)  # 0.284

    X_range = np.linspace(40, 90, 300).reshape(-1, 1)
    y_range_predict = lm1.predict(X_range)

    plt.plot(X_range, y_range_predict, color="blue")
    plt.show()


def polyModel(deg):
    poly = PolynomialFeatures(degree=deg)
    X_train_2 = poly.fit_transform(X_train)
    X_test_2 = poly.transform(X_test)

    lm2 = LinearRegression()
    lm2.fit(X_train_2, y_train)

    y_train_predict_2 = lm2.predict(X_train_2)
    y_test_predict_2 = lm2.predict(X_test_2)
    r2_score_train = r2_score(y_train, y_train_predict_2)
    r2_score_test = r2_score(y_test, y_test_predict_2)

    print("degree:", deg, " r2_score_train:", r2_score_train)  # 0.995
    print("degree:", deg, " r2_score_test:", r2_score_test)  # 0.994

    X_rang = np.linspace(20, 100, 300).reshape(-1, 1)
    X_range_2 = poly.transform(X_rang)
    y_range_2_predict = lm2.predict(X_range_2)

    plt.plot(X_rang, y_range_2_predict, color="blue")
    test = plt.scatter(X_test, y_test_predict_2, color="green")
    plt.legend((train, test), ("train", "test"))
    plt.show()


polyModel(5)
