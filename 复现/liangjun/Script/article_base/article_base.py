"""
文章对应的基本模型
"""
import keras.backend as K
import keras
import numpy as np
from keras.layers import RepeatVector, Dense, multiply, dot, Lambda, Reshape, Concatenate, LSTM, Input, Bidirectional, \
    Dropout, merge, Permute, Reshape
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import random
import time
import tensorflow as tf
from sklearn.metrics import r2_score
from keras.models import Model, Input
# import pandas as pd
from keras import optimizers

seed = 1
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

pred = [1, 4, 6]


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:i + look_back, np.delete(list(np.arange(23)), pred)]
        X.append(x)
        y = dataset[i + look_back - 1, pred]
        Y.append(y)
    return np.array(X), np.array(Y)


def create_model(look_back=50, n_output=3, feature_num=20, batch_size=64,
                 encoder_units=128,
                 decoder_units=128):
    # 共用了同一个特征提取器


    # 定义输入层
    model_input = Input(batch_shape=(batch_size, look_back, feature_num))
    # shape = [64, 50, 20]


    # 初始化上一个隐藏状态和上一个细胞状态
    h_prev = Lambda(lambda x: K.zeros([batch_size, encoder_units]))(model_input)  # 这个为啥要用Lambda；
    c_prev = Lambda(lambda x: K.zeros([batch_size, encoder_units]))(model_input)
    # shape = [64, 256]
    # 创建编码器输出列表
    encoder_output_list = []
    # 定义编码器 LSTM
    encoder_lstm = LSTM(encoder_units, return_state=True, stateful=False)  # return_state:是否返回隐藏状态和细胞状态
    #                        # stateful = False,＃如果为True，则批次中索引i的每个样本的最后状态将用作下一个批次中索引i的样本的初始状态。
    # 对输入的每个时间步执行编码
    for i in range(look_back):  # 每个输入
        # 取出当前时间步的输入
        model_input_ = Lambda(lambda x: x[:, i, :])(model_input) # shape =(64, 20)
        # 将当前时间步的输入与上一时间步的隐藏状态进行拼接
        concat = Concatenate(axis=-1)([model_input_, h_prev])
        # shape = [64, 276]
        # 经过全连接层得到能量
        energies = keras.layers.Dense(feature_num, activation="tanh")(concat)
        # shape = [64, 20]
        # 计算alpha概率
        a_probs = Lambda(lambda x: K.softmax(x, axis=-1), name='a_probs' + str(i))(energies)
        # shape = [64, 20]
        # 根据alpha概率进行编码器输入的加权求和
        encoder_input = merge.Multiply()([model_input_, a_probs])  # TODO： Step——1 ： x * a_probs
        # shape = [64, 20]
        # 对加权求和结果进行形状变换
        encoder_input = Reshape((1, feature_num))(encoder_input)
        # shape = [64, 1, 20]


        # 执行编码器 LSTM，并得到新的编码器输出、隐藏状态和细胞状态
        encoder_output_, h_prev, c_prev = encoder_lstm(encoder_input, initial_state=[h_prev, c_prev])  # TODO： Step——2 ： 保存状态，并输入LSTM encoder
        #  这里的代码 输出 h_prev, c_prev 是 LSTM 的哪个位置；
        # 将新的编码器输出添加到列表中
        encoder_output_list.append(encoder_output_)

    # 将所有时间步的编码器输出进行拼接
    encoder_output = Concatenate(axis=1)(encoder_output_list)  # TODO： Step——3 ： 变成一行
    # shape = [64, 6400]
    # 对编码器输出进行形状变换
    encoder_output = Reshape((encoder_units, look_back))(encoder_output)
    # encoder-output shape = [64,128,50]

    # 对编码器输出进行轴换序
    encoder_output = Permute((2, 1))(encoder_output)  # 好怪

    # 定义解码层 LSTM
    decoder_lstm = LSTM(decoder_units, return_state=True, stateful=False)

    # 共用了同一个特征提取器， 此处定义了 多个全连接层 ， 对应三个输出

    # 定义多个全连接层，用于输出不同特征
    dense1 = Dense(128, activation='relu', name='feature1')
    dense2 = Dense(128, activation='relu', name='feature2')
    dense3 = Dense(128, activation='relu', name='feature3')

    # 定义输出层，输出三个不同的结果
    decoder_dense1 = Dense(1, activation='sigmoid')  # 输出层
    decoder_dense2 = Dense(1, activation='sigmoid')  # 输出层
    decoder_dense3 = Dense(1, activation='sigmoid')  # 输出层

    # 初始化输出列表
    outputs = []

    # 初始化上一个解码状态和上一个解码细胞状态
    h_prev_de = Lambda(lambda x: K.zeros([batch_size, 1, decoder_units]))(model_input)
    c_prev_de = Lambda(lambda x: K.zeros([batch_size, 1, decoder_units]))(model_input)
    # 初始化解码器 LSTM 输出
    decoder_lstm_output = Lambda(lambda x: K.zeros([batch_size, decoder_units]))(model_input)

    # 对每个输出进行预测
    for t in range(n_output):
        # 将上一个解码状态重复 look_back 次，并与编码器输出进行拼接
        con = Concatenate(axis=1)([h_prev_de] * look_back)  # 要look_back 长度 的h_prev_de
        # con shape = [64, 50, 256]
        # 将拼接结果与编码器输出进行再次拼接
        concat2 = Concatenate(axis=2)([encoder_output, con])  # 拼接了
        # 经过全连接层得到能量
        energies = keras.layers.Dense(decoder_units, activation="tanh", name='e1' + str(t))(concat2)
        # 经过全连接层得到alpha概率
        energies = keras.layers.Dense(1, activation="linear", name='e2' + str(t))(energies)  # TODO： step--4； time-wise layer
        # 计算alpha概率
        alphas = Lambda(lambda x: K.softmax(x, axis=1), name='aplhas' + str(t))(energies)  #  Dense 的权重边 进行 softmax 处理

        # 根据alpha概率进行编码器输出的加权求和
        decoder_input_1 = keras.layers.dot([alphas, encoder_output], axes=1)  # 求得加权和； 输出值 是v1...vq

        # 根据输出序号选择不同的上下文输入
        if t == 0:
            decoder_input_1 = Lambda(lambda x: x, name='context1')(decoder_input_1)
        elif t == 1:
            decoder_input_1 = Lambda(lambda x: x, name='context2')(decoder_input_1)
        elif t == 2:
            decoder_input_1 = Lambda(lambda x: x, name='context3')(decoder_input_1)
        # 将解码器 LSTM 输出进行形状变换
        decoder_input_2 = Reshape((1, decoder_units))(decoder_lstm_output)
        # 将两个输入进行拼接
        decoder_input = Concatenate(axis=-1)([decoder_input_1, decoder_input_2])  #  TODO ： Step——5 ：拼接上 上一次的 FC 输出

        # 将上一个解码状态进行形状变换
        h_prev_de = Reshape((decoder_units,))(h_prev_de)
        # 将上一个解码细胞状态进行形状变换
        c_prev_de = Reshape((decoder_units,))(c_prev_de)

        # 执行解码器 LSTM，并得到新的解码器输出、隐藏状态和细胞状态
        decoder_lstm_output, h_prev_de, c_prev_de = decoder_lstm(decoder_input, initial_state=[h_prev_de, c_prev_de])

        # 将上一个解码状态进行形状变换
        h_prev_de = Reshape((1, decoder_units))(h_prev_de)
        # 将上一个解码细胞状态进行形状变换
        c_prev_de = Reshape((1, decoder_units))(c_prev_de)

        # 根据输出序号选择不同的全连接层进行预测
        if t == 0:
            decoder_lstm_output_ = dense1(decoder_lstm_output)
            dense_output = decoder_dense1(decoder_lstm_output_)   #输出层
        elif t == 1:
            decoder_lstm_output_ = dense2(decoder_lstm_output)
            dense_output = decoder_dense2(decoder_lstm_output_)
        elif t == 2:
            decoder_lstm_output_ = dense3(decoder_lstm_output)
            dense_output = decoder_dense3(decoder_lstm_output_)

        # 将每个输出添加到输出列表中
        outputs.append(dense_output)

    # 将三个输出进行拼接
    outputs = Concatenate(axis=1)(outputs)
    # 构建模型
    model = keras.Model(model_input, outputs)
    return model


batch_size = 40
look_back = 50
feature_num = 20

table = loadmat(r"D:\PythonProject\TimeSeriesWork\复现\liangjun\data\cigarette\cigarette\original data.mat")
data = table['datante']

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

X, Y = create_dataset(data, look_back)
Y = Y.squeeze()

print(X.shape, Y.shape)

trainX, testX = X[:400], X[400:560]
train_Y, test_Y = Y[:400], Y[400:560]
print("训练集数据 与 测试集数据shape")
print(trainX.shape, train_Y.shape)
print(testX.shape, test_Y.shape)

model = create_model(look_back=look_back, n_output=3, feature_num=20, batch_size=batch_size,
                     encoder_units=128,
                     decoder_units=128)
adam = optimizers.Adam(lr=2e-3)
model.compile(loss='mean_squared_error', optimizer=adam)  # loss = MSE

time_start = time.time()
for i in range(300):
    print(i)
    model.fit(trainX, train_Y, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
    model.reset_states()
testTime = time.time() - time_start
print(round(testTime, 4))

predicts = model.predict(testX, batch_size=batch_size)
print("预测的shape")
print(predicts.shape)

# 为什么将三个预测结果 都加上一定的偏置
predicts[:, 0] += 17.5
test_Y[:, 0] += 17.5

predicts[:, 1] += 11.5
test_Y[:, 1] += 11.5

predicts[:, 2] += 11.5
test_Y[:, 2] += 11.5

print("绘制三个预测结果")

plt.figure(1)
plt.plot(predicts[:, 0])
plt.plot(test_Y[:, 0])
print(np.sqrt(mean_squared_error(predicts[:, 0], test_Y[:, 0])))
print(r2_score(predicts[:, 0], test_Y[:, 0]))


plt.figure(2)
plt.plot(predicts[:, 1])
plt.plot(test_Y[:, 1])
print(np.sqrt(mean_squared_error(predicts[:, 1], test_Y[:, 1])))
print(r2_score(predicts[:, 1], test_Y[:, 1]))


plt.figure(3)
plt.plot(predicts[:, 2])
plt.plot(test_Y[:, 2])
print(np.sqrt(mean_squared_error(predicts[:, 2], test_Y[:, 2])))
print(r2_score(predicts[:, 2], test_Y[:, 2]))
plt.show()
# layers = [5,67,127,187]
#
##layers=[5,13,19,25,31,37,43,49,55,61,67,73,79,85,91,97,103,109,115,121]
#
# a_list = []
# for i in range(len(layers)):
#    plt.figure(i+3)
#    layer_model = Model(inputs=model.input,outputs=[model.layers[layers[i]].output])
#    a = layer_model.predict(testX,batch_size=batch_size)
#    a = np.mean(a,axis=0)
#    a_list.append(a)
#    a = pd.DataFrame(a)
#    plt.plot(a)
#    plt.show()
#    print(a.shape)


#
# layers = [-42,-27,-15]
# a_list = []
# for i in range(len(layers)):
#    plt.figure(i+23)
#    layer_model = Model(inputs=model.input,outputs=[model.layers[layers[i]].output])
#    a = layer_model.predict(testX,batch_size=batch_size)
#    a = np.mean(a,axis=0)
#    a_list.append(a)
#    a = pd.DataFrame(a)
#    plt.plot(a)
#    plt.show()
#    print(a.shape)
#
#
# aa = np.column_stack(a_list)
# pd.DataFrame(aa).to_excel('/Users/zhuxiaoxiansheng/Desktop/testdata_attention4.xlsx')
#
#
#
# layers = [-42,-27,-15]
# a_list = []
# for i in range(len(layers)):
#    plt.figure(i+23)
#    layer_model = Model(inputs=model.input,outputs=[model.layers[layers[i]].output])
#    a = layer_model.predict(trainX,batch_size=batch_size)
#    a = np.mean(a,axis=0)
#    a_list.append(a)
#    a = pd.DataFrame(a)
#    plt.plot(a)
#    plt.show()
#    print(a.shape)
#
#
# aa = np.column_stack(a_list)
# pd.DataFrame(aa).to_excel('/Users/zhuxiaoxiansheng/Desktop/traindata_attention4.xlsx')
#
#
#
#
#


# features = layer_model.predict(np.concatenate([trainX,testX],axis=0),batch_size)
# attention_vector_final = features[100,0,:].squeeze()
# print(attention_vector_final.shape)
# a = pd.DataFrame(attention_vector_final)
# plt.plot(attention_vector_final)
# plt.show()
# a.to_excel('/Users/zhuxiaoxiansheng/Desktop/449sample.xlsx')


# feature1 = features[0].squeeze()
# feature2 = features[1].squeeze()
# feature3 = features[2].squeeze()
#
# from sklearn import manifold
# from mpl_toolkits.mplot3d import Axes3D
#
#
# tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
#
# feature1 = tsne.fit_transform(feature1)
# test_min, test_max = feature1.min(0), feature1.max(0)
# feature1 = (feature1 - test_min) / (test_max - test_min)  # 归一化
#
# feature2 = tsne.fit_transform(feature2)
# test_min, test_max = feature2.min(0), feature2.max(0)
# feature2 = (feature2 - test_min) / (test_max - test_min)  # 归一化
#
# feature3 = tsne.fit_transform(feature3)
# test_min, test_max = feature3.min(0), feature3.max(0)
# feature3 = (feature3 - test_min) / (test_max - test_min)  # 归一化
#
#
# plt.rcParams['savefig.dpi'] = 600 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
#
# fig = plt.figure(figsize=(7, 7))
# ax = Axes3D(fig)
#
# for i in range(feature1.shape[0]):
#    ax.scatter(feature1[i, 0],feature1[i, 1],feature1[i, 2],s=3,c='red')
# for i in range(feature2.shape[0]):
#    ax.scatter(feature2[i, 0],feature2[i, 1],feature2[i, 2],s=3,c='yellow')
# for i in range(feature3.shape[0]):
#    ax.scatter(feature3[i, 0],feature3[i, 1],feature3[i, 2],s=3,c='blue')
#
#
##ax.grid(False)
##plt.axis('off')
# plt.show()
# fig.savefig('/Users/zhuxiaoxiansheng/Desktop/图4.png', dpi=300)
#

#
# pre = model.predict(testX,batch_size=batch_size)
# predicts = pre.squeeze()
# print(predicts.shape)
#
# predicts[:,0] += 17.5
# test_Y[:,0] += 17.5
# predicts[:,1] += 11.5
# test_Y[:,1] += 11.5
# predicts[:,2] += 11.5
# test_Y[:,2] += 11.5
#
##plt.figure(1)
##plt.plot(predicts[:,0])
##plt.plot(test_Y[:,0])
# print(np.sqrt(mean_squared_error(predicts[:,0],test_Y[:,0])))
# print(r2_score(predicts[:,0],test_Y[:,0]))
##plt.figure(2)
##plt.plot(predicts[:,1])
##plt.plot(test_Y[:,1])
# print(np.sqrt(mean_squared_error(predicts[:,1],test_Y[:,1])))
# print(r2_score(predicts[:,1],test_Y[:,1]))
##plt.figure(3)
##plt.plot(predicts[:,2])
##plt.plot(test_Y[:,2])
# print(np.sqrt(mean_squared_error(predicts[:,2],test_Y[:,2])))
# print(r2_score(predicts[:,2],test_Y[:,2]))
#
#


# import pandas as pd
# predicts = pd.DataFrame(predicts)
# test_Y = pd.DataFrame(test_Y.squeeze())
# predicts.to_excel('/Users/zhuxiaoxiansheng/Desktop/S2S/卷烟对比实验/attention_res2.xlsx')
# test_Y.to_excel('/Users/zhuxiaoxiansheng/Desktop/S2S/卷烟对比实验/test_Y.xlsx')
#


#
# pre = [1,4,6]
# print()
# traindata = data[50:450,np.delete(list(np.arange(23)),pre)]
# trainlabel = data[50:450,pre]
# testdata = data[450:,np.delete(list(np.arange(23)),pre)]
# testlabel = data[450:,pre]
#
# from sklearn.ensemble import RandomForestRegressor
# clf = RandomForestRegressor()
# clf.fit(traindata,trainlabel[:,0])
# res = clf.predict(testdata)
# print(np.sqrt(mean_squared_error(res,testlabel[:,0])))
# print(r2_score(res,testlabel[:,0]))
# clf = RandomForestRegressor()
# clf.fit(traindata,trainlabel[:,1])
# res = clf.predict(testdata)
# print(np.sqrt(mean_squared_error(res,testlabel[:,1])))
# print(r2_score(res,testlabel[:,1]))
# clf = RandomForestRegressor()
# clf.fit(traindata,trainlabel[:,2])
# res = clf.predict(testdata)
# print(np.sqrt(mean_squared_error(res,testlabel[:,2])))
# print(r2_score(res,testlabel[:,2]))
#
#
#
