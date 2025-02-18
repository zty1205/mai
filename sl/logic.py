# 分类问题

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("logic.csv")
# print(data.head())

x = data.drop(columns=["Pass"])
x1 = data.loc[:, "Exam1"]
x2 = data.loc[:, "Exam2"]
# print(x, type(x))
y = data.loc[:, "Pass"]


def One():
    model = LogisticRegression()
    model.fit(x, y)
    # y_predicted = model.predict(x)
    # print(y_predicted)
    # print("均方误差：", mean_squared_error(y, y_predicted))
    # print("R2：", r2_score(y, y_predicted))

    # x_test = [[70, 65]]
    # y_test = model.predict(x_test)
    # print("passed" if y_test[0] == 1 else "failed")

    theta = model.intercept_
    # print(theta)
    theta1 = model.coef_[0][0]
    theta2 = model.coef_[0][1]

    x1_new = -(theta + theta1 * x1) / theta2
    x2_new = -(theta + theta2 * x2) / theta1
    # print(x1_new)

    # fig1 = plt.figure()
    # mask = data["Pass"] == 1
    # passed = plt.scatter(
    #     data.loc[:, "Exam1"][mask], data.loc[:, "Exam2"][mask], color="blue"
    # )
    # failed = plt.scatter(
    #     data.loc[:, "Exam1"][~mask], data.loc[:, "Exam2"][~mask], color="red"
    # )
    # plt.xlabel("Exam1")
    # plt.ylabel("Exam2")
    # plt.legend((passed, failed), ("Passed", "Failed"))
    # plt.plot(x1, x1_new, color="green")
    # plt.plot(x2, x2_new, color="blue")
    # plt.show()


# One()

# 二阶边界函数

x1_2 = x1 * x1
x2_2 = x2 * x2
x1_x2 = x1 * x2


def Two():
    x_new = {
        "X1": x1,
        "X2": x2,
        "X1_2": x1_2,
        "X2_2": x2_2,
        "x1_x2": x1_x2,
    }
    x_new = pd.DataFrame(x_new)
    # print(x_new)
    model = LogisticRegression()
    model.fit(x_new, y)
    y_predicted = model.predict(x_new)
    # print("均方误差：", mean_squared_error(y, y_predicted))
    # print("R2：", r2_score(y, y_predicted))
    print(model.intercept_)

    x1_new = x1.sort_values()

    theta0 = model.intercept_
    theta1, theta2, theta3, theta4, theta5 = (
        model.coef_[0][0],
        model.coef_[0][1],
        model.coef_[0][2],
        model.coef_[0][3],
        model.coef_[0][4],
    )
    # print(theta, theta1, theta2, theta3, theta4, theta5)
    a = theta4
    b = theta5 * x1_new + theta2
    c = theta0 + theta1 * x1_new + theta3 * x1_new * x1_new
    x2_new_boundary = (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)

    fig1 = plt.figure()
    mask = data["Pass"] == 1
    passed = plt.scatter(
        data.loc[:, "Exam1"][mask], data.loc[:, "Exam2"][mask], color="blue"
    )
    failed = plt.scatter(
        data.loc[:, "Exam1"][~mask], data.loc[:, "Exam2"][~mask], color="red"
    )
    plt.xlabel("Exam1")
    plt.ylabel("Exam2")
    plt.legend((passed, failed), ("Passed", "Failed"))
    plt.plot(x1_new, x2_new_boundary, color="green")
    plt.show()


Two()
