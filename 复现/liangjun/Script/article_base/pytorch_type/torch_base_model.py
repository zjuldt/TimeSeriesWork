"""
准备使用torch 进行模型复现
"""
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, look_back=50, n_output=3, feature_num=20, batch_size=64, encoder_units=128, decoder_units=128):
        super(BaseModel, self).__init__()

        # params
        self.look_back = look_back
        self.n_output = n_output
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

        # 初始化LSTM的输入
        # h_prev = torch.zeros(batch_size, encoder_units)
        # c_prev = torch.zeros(batch_size, encoder_units)

        # models

        self.encoder_LSTM = nn.LSTM(feature_num, encoder_units, batch_size)  # 输出 ： output, h, c
        # torch 中的调用 （input_size , hidden_size, num_layer）
        self.dense_list = nn.ModuleList([nn.Linear(feature_num + encoder_units, feature_num)
                                         for i in range(self.look_back)])

        self.decoder_LSTM = nn.LSTM(256, decoder_units, batch_size)

        # 定了三个全连接层
        self.dense_fc_list = nn.ModuleList([nn.Linear(128, 64) for i in range(self.n_output)])

        self.decoder_dense_list = nn.ModuleList([nn.Linear(64, 1) for i in range(self.n_output)])

        self.dense_list_dense = nn.ModuleList([nn.Linear(256, 128) for i in range(self.n_output)])
        self.dense_list_linear = nn.ModuleList([nn.Linear(128, 1) for i in range(self.n_output)])

    def forward(self, x):
        # Step 1: a * x

        # 初始化LSTM的输入
        h_prev = torch.zeros(self.batch_size, self.encoder_units)
        c_prev = torch.zeros(self.batch_size, self.encoder_units)

        encoder_output_list = []
        for i in range(self.look_back):
            input_item = x[:, i, :]
            concat = torch.cat((input_item, h_prev), dim=1)
            # concat shape = [64, 128 + 20] = [64, 148]
            energies = self.dense_list[i](concat)
            energies = torch.tanh(energies)
            # energies shape = [64, 20]
            a_probs = nn.functional.softmax(energies, dim=-1)
            # a_probs shape = [64, 20]

            encode_input = torch.mul(a_probs, input_item)
            # encode-input shape = [64, 20]
            encode_input = torch.reshape(encode_input, (self.batch_size, self.feature_num))
            # encode-input shape = [64, 20]
            encode_output_, (h_prev, c_prev) = self.encoder_LSTM(encode_input, [h_prev, c_prev])
            encoder_output_list.append(encode_output_)

        encoder_output = torch.cat(encoder_output_list, dim=1)
        # 在encoder_output_list中，每个元素的shape = [64, 128]
        # encoder_output shape = [64, 6400]

        encoder_output = torch.reshape(encoder_output, (self.batch_size, self.encoder_units, self.look_back))
        encoder_output = torch.transpose(encoder_output, 2, 1)
        # encoder_output shape = [64, 50, 128]

        outputs = []

        h_prev_de = torch.zeros(self.batch_size, 1, self.decoder_units)
        c_prev_de = torch.zeros(self.batch_size, 1, self.decoder_units)

        decoder_lstm_output = torch.zeros(self.batch_size, self.decoder_units)
        for i in range(self.n_output):
            con = torch.cat([h_prev_de] * self.look_back, dim=1)
            # con shape = [64, 50 , 128]
            concat2 = torch.cat([encoder_output, con], dim=2)
            # concat2 shape = [64, 50, 256]
            energies = self.dense_list_dense[i](concat2)
            energies = torch.tanh(energies)  # activate function : tanh
            # energies shape = [64, 50, 128]
            energies = self.dense_list_linear[i](energies)  # activate function : linear

            # energies shape = [64, 50, 1]
            alpha = nn.functional.softmax(energies, dim=1)
            # alpha shape = [64, 50 , 1]
            #### 解决dot问题
            #  torch.matmul(A.permute(0, 2, 1), B)
            # decoder_input_1 = torch.mul(alpha, encoder_output)
            decoder_input_1 = torch.matmul(alpha.permute(0, 2, 1), encoder_output)
            #                          [64, 50 , 1]   [64, 50, 128] -> [64, 1, 128]
            # decoder_input_1 shape = [64, 1, 128]
            decoder_input_2 = torch.reshape(decoder_lstm_output, (self.batch_size, 1, self.decoder_units))
            # decoder_input_2 shape = [ 64, 1, 128]
            decoder_input = torch.cat([decoder_input_1, decoder_input_2], dim=-1)
            # decoder_input shape = [64, 1, 256]

            h_prev_de = torch.reshape(h_prev_de, (self.batch_size, 1, self.decoder_units))
            c_prev_de = torch.reshape(c_prev_de, (self.batch_size, 1, self.decoder_units))

            decoder_output, (h_prev_de, c_prev_de) = self.decoder_LSTM(decoder_input, [h_prev_de, c_prev_de])

            h_prev_de = torch.reshape(h_prev_de, (self.batch_size, 1, self.decoder_units))
            c_prev_de = torch.reshape(c_prev_de, (self.batch_size, 1, self.decoder_units))

            decoder_lstm_output_ = self.dense_fc_list[i](decoder_lstm_output)
            decoder_lstm_output_ = torch.tanh(decoder_lstm_output_)
            decoder_output = self.decoder_dense_list[i](decoder_lstm_output_)
            decoder_output = torch.sigmoid(decoder_output)

            outputs.append(decoder_output)
        return torch.cat(outputs, dim=1)
