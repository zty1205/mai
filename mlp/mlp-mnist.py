from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

def load_mnist_images(image_path):
    """加载 MNIST 图像文件"""
    with open(image_path, 'rb') as f:
        # 读取二进制数据
        data = np.fromfile(f, dtype=np.uint8, offset=16)  # 跳过前16个字节的头部信息
    return data.reshape(-1, 28, 28)  # 重塑为 28x28 的图像

def load_mnist_labels(label_path):
    """加载 MNIST 标签文件"""
    with open(label_path, 'rb') as f:
        # 读取二进制数据
        data = np.fromfile(f, dtype=np.uint8, offset=8)  # 跳过前8个字节的头部信息
    return data
  
x_train = load_mnist_images('train-images-idx3-ubyte')
y_train = load_mnist_labels('train-labels-idx1-ubyte')
x_test = load_mnist_images('t10k-images-idx3-ubyte')
y_test = load_mnist_labels('t10k-labels-idx1-ubyte')
# print(x_train.shape, y_train.shape)
  
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

img1 = x_train[0] # 28x28
# fig1 = plt.figure()
# plt.imshow(img1)
# plt.show()

feature_size = img1.shape[0] * img1.shape[1]
x_train_format = x_train.reshape(x_train.shape[0], feature_size) # 60000x784
x_test_format = x_test.reshape(x_test.shape[0], feature_size)

# 归一化，源数据是0-255，归一化后是0-1
x_train_format = x_train_format / 255
x_test_format = x_test_format / 255

from keras.utils import to_categorical

# 转换输出结果格式 y的分类转成矩阵
y_train_format = to_categorical(y_train)
y_test_format = to_categorical(y_test)

mlp = Sequential()
mlp.add(Dense(units=392, input_dim=feature_size, activation='sigmoid'))
mlp.add(Dense(units=392, activation='sigmoid'))
mlp.add(Dense(units=10, activation='softmax'))

mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mlp.fit(x_train_format, y_train_format, epochs=10)

y_pred_probs = mlp.predict(x_test_format)
y_test_predict = np.argmax(y_pred_probs, axis=1)

# 预测第一个数据
img2 = x_test[0]
fig2 = plt.figure()
plt.imshow(img2)
plt.title("Predict: %d" % y_test_predict[0])
plt.show()


