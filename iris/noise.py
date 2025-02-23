# 生成x = 6-11, y = 7-14的数据, 数据量为100，则保留1位小数
# 添加4个噪声点，分别为(0,5, 20), (16, 0.5), (8,5), (11.2, 14.1)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.covariance import EllipticEnvelope

# 读取数据，并绘制散点图
data = pd.read_csv("iris/noise.csv")

x1 = data.loc[:, "x"]
x2 = data.loc[:, "y"]

# plt.figure(figsize=(200, 200))
fig1 = plt.subplot(221)
plt.scatter(x1, x2, color="blue")
plt.xlabel("x1")
plt.ylabel("x2")
# plt.show()

fig2 = plt.subplot(222)
plt.hist(x1, bins=20, color="blue", alpha=0.5)

fig3 = plt.subplot(223)
plt.hist(x2, bins=20, color="red", alpha=0.5)

# plt.show()

# 计算均值和标准差
x1_mean = np.mean(x1)
x1_std = np.std(x1)
# x2_mean = np.mean(x2)
# x2_std = np.std(x2)

x1_range = np.linspace(0, 20, 300)
normal_1 = norm.pdf(x1_range, x1_mean, x1_std)

fig3 = plt.subplot(224)
plt.plot(x1_range, normal_1, color="blue")
# plt.show()

# 使用默认的阈值时0.1 有些正常的值被认为是异常值
clf = EllipticEnvelope(contamination=0.02)
clf.fit(data)

y_predict = clf.predict(data)

anamoly_points = plt.scatter(
    x1[y_predict == 1],
    x2[y_predict == 1],
    marker="o",
    facecolor="none",
    edgecolor="red",
    s=250,
)


def plot():
    fig = plt.figure(figsize=(12, 8))
    plt.scatter(x1, x2, color="blue")
    plt.scatter(x1[y_predict == -1], x2[y_predict == -1], color="red")
    plt.show()


plot()
