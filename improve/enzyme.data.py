# 生成一个 x在40-90， y在0-3的描述酶活性的二维数组， x为温度，y为活性，
# xy的值保留2位小数, xy的关系满足一元二次方程，必须包含点[40, 1,8]，[60, 3], [90, 0.5] 并画出散点图

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 解方程组
A = np.array([[1600, 40, 1], [3600, 60, 1], [8100, 90, 1]])
B = np.array([1.8, 3, 0.5])
coefficients = np.linalg.solve(A, B)
a, b, c = coefficients

# 生成数据
x_values = np.linspace(40, 90, 100)
x_error_std = 0.0492
x_values = np.round(x_values, 6) + np.random.normal(0, x_error_std, x_values.shape)
print(x_values)
y_values = a * x_values**2 + b * x_values + c

# 添加随机误差
error_std = 0.147
y_values_with_error = y_values + np.random.normal(0, error_std, y_values.shape)

# 保留两位小数
y_values_with_error = np.round(y_values_with_error, 2)

# 画出散点图
plt.scatter(x_values, y_values_with_error, label="Generated Data with Error")
plt.scatter([40, 60, 90], [1.8, 3, 0.5], color="red", label="Given Points")
plt.xlabel("Temperature (x)")
plt.ylabel("Activity (y)")
plt.legend()
# plt.show()

# 将数据保存为# enzyme.csv文件
df = pd.DataFrame({"Temperature": x_values, "Activity": y_values_with_error})
df.to_csv("enzyme.csv", index=False)
