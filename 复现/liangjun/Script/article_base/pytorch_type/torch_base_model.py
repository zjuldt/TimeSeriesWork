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
        h_prev = torch.zeros(batch_size, encoder_units)
        c_prev = torch.zeros(batch_size, encoder_units)

        # models

        self.encoder_LSTM = nn.LSTM(encoder_units)  # 输出 ： output, h, c

        self.dense_list = [nn.Linear(feature_num + encoder_units, feature_num, activation=nn.tanh())] * self.look_back

        self.decoder_LSTM = nn.LSTM(decoder_units)

        # 定了三个全连接层
        self.dense_fc_list = [nn.Linear(64, 128, activation=nn.tanh())] * self.n_output

        self.decoder_dense_list = [nn.Linear(64, 1, activation=nn.tanh())] * self.n_output

        self.dense_list_dense = [nn.Linear(feature_num + encoder_units, decoder_units,
                                           activation=nn.tanh())] * self.n_output
        self.dense_list_linear = [nn.Linear(decoder_units, 1, activation=nn.linear())] * self.n_output

    def forward(self, x):
        # Step 1: a * x

        # 初始化LSTM的输入
        h_prev = torch.zeros(self.batch_size, self.encoder_units)
        c_prev = torch.zeros(self.batch_size, self.encoder_units)

        encoder_output_list = []
        for i in range(self.look_back):
            input_item = x[:, i, :]
            concat = torch.cat(input_item, h_prev, dim=1)
            energies = self.dense_list[i](concat)
            a_probs = nn.functional.softmax(energies, dim=1)
            encode_input = torch.mul(a_probs, input_item)
            encode_input = torch.reshape(encode_input, (self.batch_size, 1, self.feature_num))

            encode_output_, h_prev, c_prev = self.encoder_LSTM(encode_input, (h_prev, c_prev))
            encoder_output_list.append(encode_output_)

        encoder_output = torch.cat(encoder_output_list, dim=1)
        encoder_output = torch.reshape(encoder_output, (self.batch_size, self.look_back, self.feature_num))
        encoder_output = torch.transpose(encoder_output, 0, 1)

        outputs = []

        h_prev_de = torch.zeros(self.batch_size, 1, self.decoder_units)
        c_prev_de = torch.zeros(self.batch_size, 1, self.decoder_units)

        decoder_lstm_output = torch.zeros(self.batch_size, self.decoder_units)
        for i in range(self.n_output):
            con = torch.cat([h_prev_de] * self.look_back)
            concat2 = torch.cat([encoder_output, con], dim=2)
            energies = self.dense_list_dense[i](concat2)
            energies = self.dense_list_linear[i](energies)

            alpha = nn.functional.softmax(energies, dim=1)
            decoder_input_1 = torch.mul(alpha, encoder_output)

            decoder_input_2 = torch.reshape(decoder_lstm_output, (self.batch_size, 1, self.decoder_units))

            decoder_input = torch.cat([decoder_input_1, decoder_input_2], dim=1)

            h_prev_de = torch.reshape(h_prev_de, (self.batch_size, 1, self.decoder_units))
            c_prev_de = torch.reshape(c_prev_de, (self.batch_size, 1, self.decoder_units))

            decoder_output, h_prev_de, c_prev_de = self.decoder_LSTM(decoder_input, (h_prev_de, c_prev_de))

            h_prev_de = torch.reshape(h_prev_de, (self.batch_size, 1, self.decoder_units))
            c_prev_de = torch.reshape(c_prev_de, (self.batch_size, 1, self.decoder_units))

            decoder_lstm_output_ = self.dense_fc_list[i](decoder_lstm_output)
            decoder_output = self.decoder_dense_list[i](decoder_lstm_output_)

            outputs.append(decoder_output)
        return torch.cat(outputs, dim=1)
