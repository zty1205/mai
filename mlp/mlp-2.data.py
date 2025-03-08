
# 给定x11, x12,  x21, x22, y, 生成100个数据集[x1, x2, y], 其中x1 = [x11, x12], x2 = [x21, x22]， x1,x2均为浮点数，保留2位小数, y为0或1
import random
import numpy as np
import pandas as pd


def generate_data(x11, x12, x21, x22, y):
    data = []
    for i in range(100):
        x1 = round(random.uniform(x11, x12), 2)
        x2 = round(random.uniform(x21, x22), 2)
        data.append([x1, x2, y])
    return data
  
  
data1 = generate_data(0, 0.44, 0, 0.44, 0)
data2 = generate_data(0.52, 1, 0, 0.44, 1)
data3 = generate_data(0, 0.44, 0.56, 1, 1)
data4 = generate_data(0.52, 1, 0.56, 1, 0)

# 将data1, data2, data3, data4合并成一个数据集，并生成CSV文件
data = data1 + data2 + data3 + data4
data = pd.DataFrame(data, columns=["x1", "x2", "y"])
data.to_csv("mlp-2.data.csv", index=False)
print("CSV数据已写入文件")
