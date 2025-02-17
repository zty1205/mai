# 线性多因子

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("mutil.csv")
print(data.head())


# 先来个单因子


def single_factor():
    # 单因子回归
    x = data.loc[:, "Income"]
    x = np.array(x).reshape(-1, 1)
    y = data.loc[:, "Price"]
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_predict = model.predict(x)
    # print("系数：", model.coef_)
    # print("截距：", model.intercept_)
    # print("均方误差：", mean_squared_error(y, y_predict))
    # print("R2：", r2_score(y, y_predict))
    fig = plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color="blue")
    plt.scatter(x, y_predict, color="red")
    plt.show()


# single_factor()

# 多因子


def multi_factor():
    x = data.drop(columns=["Price"])
    y = data.loc[:, "Price"]
    y = np.array(y).reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    y_predict = model.predict(x)
    # x_test = np.array([65000, 5, 5, 30000, 200]).reshape(1, -1)
    # y_test = model.predict(x_test)
    # print("预测值：", y_test)
    # print("系数：", model.coef_)
    # print("截距：", model.intercept_)
    # print("均方误差：", mean_squared_error(y, y_predict))
    # print("R2：", r2_score(y, y_predict))
    # fig = plt.figure(figsize=(10, 6))
    # plt.scatter(y, y_predict, color="blue")
    # plt.show()


multi_factor()
