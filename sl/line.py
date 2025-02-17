import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("line.csv")  # 10个数据  y = 2x + 5

x = data.loc[:, "x"]  # 获取x列数据
y = data.loc[:, "y"]

# plt.figure(figsize=(10, 5))
# plt.scatter(x, y, color="red")
# plt.show()

# reshape 将一维数组转换为二维数组
x = np.array(x).reshape(-1, 1)  # 10行1列
y = np.array(y).reshape(-1, 1)

model = LinearRegression()
# 拟合
model.fit(x, y)

# x_3 = [[3.5]]
# y_3 = model.predict(x_3)

y_predict = model.predict(x)

# plt.scatter(x, y, color="red")
# plt.plot(x, model.predict(x), color="blue")
# plt.show()

a = model.coef_
b = model.intercept_

# print("y = %.2fx + %.2f" % (a[0][0], b[0]))

MSE = mean_squared_error(y, y_predict)
R2 = r2_score(y, y_predict)

print("MSE: %.2f" % MSE)
print("R2: %.2f" % R2)
