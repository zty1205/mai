import random

# 定义三个区域的中心点
centers = [(0, 20), (40, 60), (70, -10)]

# 生成200个随机数据点，数据主要集中在三个区域
data = []
for _ in range(200):
    center = random.choice(centers)
    index = centers.index(center)
    x = int(center[0] + random.uniform(-10, 10))
    y = int(center[1] + random.uniform(-10, 10))
    data.append((x, y, index))

# data按index排序
data.sort(key=lambda x: x[2])
# 将数据写入CSV格式
csv_data = "\n".join([f"{x},{y},{i}" for x, y, i in data])
with open("ul/data.csv", "w") as f:
    f.write(csv_data)
print("CSV数据已写入文件")
