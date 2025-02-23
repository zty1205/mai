# 生成x = 6-11, y = 7-14的数据, 数据量为100，则保留1位小数
# 添加4个噪声点，分别为(0,5, 20), (16, 0.5), (8,5), (11.2, 14.1)
import numpy as np
import pandas as pd


def generate_data():
    x = np.round(np.random.rand(100) * 5 + 6, 1)
    y = np.round(np.random.rand(100) * 7 + 7, 1)
    data = pd.DataFrame({"x": x, "y": y})
    noise = pd.DataFrame({"x": [0, 16, 8, 11.2], "y": [5, 0.5, 5, 14.1]})
    data = pd.concat([data, noise], ignore_index=True)
    data.to_csv("noise.csv", index=False)


generate_data()
