# 根据用户 学习动力，能力提升意愿，兴趣度，空余时间，判断是否适合学习某个课程
# 逻辑回归模型使用的是 一个线性回归模型，将特征值带入到sigmoid函数中，得到的值大于0.5则为1，小于0.5则为0
# 决策树模型使用的是 一个树形结构，将特征值带入到树中，根据特征值的大小，逐步判断，最终得到结果


# 决策树的三种特征选择算法：ID3，C4.5，CART

from sklearn.datasets import load_iris
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

# 加载数据集
iris = load_iris()

# 转换为 Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

y = iris_df["target"]
X = iris_df.drop(columns=["target"], axis=1)

# 查看数据
# print(iris_df.head())
# print(iris_df['target'].value_counts())

dc_tree = tree.DecisionTreeClassifier(criterion="entropy", min_samples_leaf=5)
dc_tree.fit(X, y)

# 绘制决策树
plt.figure(figsize=(12, 8))
tree.plot_tree(
    dc_tree,
    filled=True,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
)
plt.show()
