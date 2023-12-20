# 调用torch 模型进行训练
import random

import torch
import torch.nn as nn
from torch_base_model import BaseModel
from scipy.io import loadmat
import numpy as np

# 论文模型
model = BaseModel()
print(model)
# 加载数据
batch_size = 64
look_back = 50
feature_num = 20

table = loadmat(r"D:\PythonProject\TimeSeriesWork\复现\liangjun\data\cigarette\cigarette\original data.mat")
data = table['datante']
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

pred = [1, 4, 6]


# 划分训练集和测试集
def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:i + look_back, np.delete(list(np.arange(23)), pred)]
        X.append(x)
        y = dataset[i + look_back - 1, pred]
        Y.append(y)
    return np.array(X), np.array(Y)


X, Y = create_dataset(data, look_back)
Y = Y.squeeze()

print(X.shape, Y.shape)
# x_shape = [560, 50, 20]
# y_shape = [560, 3]


trainX, testX = X[:400], X[400:560]
train_Y, test_Y = Y[:400], Y[400:560]
print("训练集数据 与 测试集数据shape")
print(trainX.shape, train_Y.shape)
# train = [400, 50, 20]
print(testX.shape, test_Y.shape)
# test = [160, 50, 20]

# train
adam = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.MSELoss()

train_loss = []
valid_loss = []

for i in range(300):
    model.train()
    train_epoch_loss = []
    if i % 3 == 0:
        print("epoch:", i)
    for j in range(len(trainX) // batch_size):
        batch_x = torch.from_numpy(trainX[j * batch_size:(j + 1) * batch_size]).float()
        batch_y = torch.from_numpy(train_Y[j * batch_size:(j + 1) * batch_size]).float()
        adam.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        adam.step()
        train_epoch_loss.append(loss.item())
        print(loss.item())
        train_loss.append(loss.item())

    model.eval()
    if i % 10 == 0:
        valid_epoch_loss = []

        for j in range(len(testX) // batch_size):
            batch_x = torch.from_numpy(testX[j * batch_size:(j + 1) * batch_size]).float()
            batch_y = torch.from_numpy(test_Y[j * batch_size:(j + 1) * batch_size]).float()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
