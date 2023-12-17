# 调用torch 模型进行训练
import random

import torch
import torch.nn as nn
from torch_base_model import BaseModel
from scipy.io import loadmat
import numpy as np

# 论文模型
model = BaseModel()

# 加载数据
batch_size = 64
look_back = 50
feature_num = 20

table = loadmat(r"/复现/liangjun/data/cigarette/cigarette/original data.mat")
data = table['datante']
seed = 1

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

trainX, testX = X[:400], X[400:560]
train_Y, test_Y = Y[:400], Y[400:560]
print("训练集数据 与 测试集数据shape")
print(trainX.shape, train_Y.shape)
print(testX.shape, test_Y.shape)

# train
adam = torch.optim.Adam(model.parameters(), lr=2e-3)
criterion = nn.MSELoss()

train_loss = []
valid_loss = []

for i in range(300):
    model.train()
    train_epoch_loss = []
    if i % 10 == 0:
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
        train_loss.append(loss.item())

    model.eval()

    valid_epoch_loss = []

    for j in range(len(testX) // batch_size):
        batch_x = torch.from_numpy(testX[j * batch_size:(j + 1) * batch_size]).float()
        batch_y = torch.from_numpy(test_Y[j * batch_size:(j + 1) * batch_size]).float()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        valid_epoch_loss.append(loss.item())
        valid_loss.append(loss.item())





