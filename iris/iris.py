from sklearn.datasets import load_iris
import pandas as pd

# 加载数据集
iris = load_iris()

# 转换为 Pandas DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df["target"] = iris.target

# 查看数据
# print(iris_df.head())
print(iris_df["target"].value_counts())
