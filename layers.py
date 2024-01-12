import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import v2
from torchsummary import summary


class EncConv(nn.Module):
    def __init__(self, h, w, ch_in, ch_out, kern, stri, pad, is_normalised=True):
        super().__init__()
        self.h_out = int(h / 2)
        self.w_out = int(w / 2)
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.kern = kern
        self.stri = stri
        self.pad = pad
        self.is_normalised = is_normalised
        self.conv = nn.Conv2d(in_channels=self.ch_in, out_channels=self.ch_out, kernel_size=self.kern, stride=self.stri,
                              padding=self.pad)
        self.ln = nn.LayerNorm([self.ch_out, h, w], elementwise_affine=False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv(inputs)
        if self.is_normalised:
            x = self.ln(x)
        x = self.relu(x)
        return x


class DownSampling(nn.Module):
    def __init__(self, ch_in, ch_out, stri, to_concat=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.stri = stri
        self.to_concat = to_concat
        self.mp = nn.MaxPool2d(kernel_size=self.stri, stride=self.stri)
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, inputs):
        x = self.mp(inputs)
        if self.to_concat:
            x = self.conv(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, ch_in, ch_out, factor, to_concat=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.factor = factor
        self.to_concat = to_concat
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=self.factor)
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, inputs):
        x = self.up_sample(inputs)
        if self.to_concat:
            x = self.conv(x)
        return x


class DecAggregator(nn.Module):
    def __init__(self, ch_in_list, ch_out_list):
        super().__init__()
        LAYER_NUM = 5
        self.ch_in_list = ch_in_list
        self.ch_out_list = ch_out_list

        # for tensors from upper encoder layers to be downsampled
        # expects to receive layers in the order like X1en, X2en, X3en ...
        down_layers = []
        for ch_in, ch_out, stri_idx in zip(ch_in_list, ch_out_list, range(len(ch_in_list), 0, -1)):
            stri = int(2 ** stri_idx)
            down_layers.append(DownSampling(ch_in=ch_in, ch_out=ch_out, stri=stri, to_concat=True))
        self.down_layers = nn.ModuleList(down_layers)

        # for tensor with the same spatial shape
        self.conv_for_same = nn.Conv2d(in_channels=int(ch_in_list[-1] * 2) if len(ch_in_list) > 0 else 64,
                                       out_channels=64, kernel_size=3, stride=1, padding=1)

        # for tensor from lower decoder layers to be upsampled
        # expects to receive layers in the order like X5en, X4de, X3de ...
        up_layers = []
        up_layers_num = LAYER_NUM - (len(ch_in_list) + 1)
        for i in range(up_layers_num):
            ch_in = 64 * (2 ** (LAYER_NUM - 1 - i))
            factor = 2 ** (up_layers_num - i)
            up_layers.append(UpSampling(ch_in=ch_in, ch_out=64, factor=factor, to_concat=True))
        self.up_layers = nn.ModuleList(up_layers)

        # concat and convolution
        channel = LAYER_NUM * 64
        self.conv_for_last = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()

    def forward(self, enc_list, enc_same, dec_list):
        tensor_list = []
        for layer, enc in zip(self.down_layers, enc_list):
            tensor_list.append(layer(enc))

        tensor_list.append(self.conv_for_same(enc_same))

        for layer, dec in zip(self.up_layers, dec_list):
            tensor_list.append(layer(dec))

        tensor = torch.concatenate(tensor_list, axis=1)
        tensor = self.conv_for_last(tensor)
        tensor = self.bn(tensor)
        tensor = self.relu(tensor)
        #         print(f"\t\t\t{tensor.size() = }")
        return tensor


class HeadForSupervision(nn.Module):
    def __init__(self, ch_in, ch_out, size):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)
        self.up_sample = nn.UpsamplingBilinear2d(size=size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.up_sample(x)
        x = self.sigmoid(x)
        return x


class HeadForClassification(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.dropout = nn.Dropout()
        self.conv = nn.Conv2d(ch_in, ch_out, 1, 1)
        self.amp = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.dropout(inputs)
        x = self.conv(x)
        x = self.amp(x)
        x = self.sigmoid(x)

        x = x.squeeze(-1).squeeze(-1)
        x = x.argmax(dim=1)
        x = x.unsqueeze(1).float()
        return x


class UNet3Plus(nn.Module):
    def __init__(self, num_classes, resized_size):
        super().__init__()
        self.resized_size = resized_size
        self.num_classes = num_classes
        self.en_1 = EncConv(h=Config.H, w=Config.W, ch_in=3, ch_out=64, kern=3, stri=1, pad=1)
        self.en_2 = EncConv(h=self.en_1.h_out, w=self.en_1.w_out, ch_in=self.en_1.ch_out, ch_out=self.en_1.ch_out * 2,
                            kern=3, stri=1, pad=1)
        self.en_3 = EncConv(h=self.en_2.h_out, w=self.en_2.w_out, ch_in=self.en_2.ch_out, ch_out=self.en_2.ch_out * 2,
                            kern=3, stri=1, pad=1)
        self.en_4 = EncConv(h=self.en_3.h_out, w=self.en_3.w_out, ch_in=self.en_3.ch_out, ch_out=self.en_3.ch_out * 2,
                            kern=3, stri=1, pad=1)
        self.en_5 = EncConv(h=self.en_4.h_out, w=self.en_4.w_out, ch_in=self.en_4.ch_out, ch_out=self.en_4.ch_out * 2,
                            kern=3, stri=1, pad=1)
        self.en_1_down = DownSampling(ch_in=self.en_1.ch_out, ch_out=self.en_1.ch_out, stri=2)
        self.en_2_down = DownSampling(ch_in=self.en_2.ch_out, ch_out=self.en_2.ch_out, stri=2)
        self.en_3_down = DownSampling(ch_in=self.en_3.ch_out, ch_out=self.en_3.ch_out, stri=2)
        self.en_4_down = DownSampling(ch_in=self.en_4.ch_out, ch_out=self.en_4.ch_out, stri=2)
        self.en_5_down = DownSampling(ch_in=self.en_5.ch_out, ch_out=self.en_5.ch_out, stri=2)

        self.de_4 = DecAggregator([self.en_1_down.ch_out, self.en_2_down.ch_out, self.en_3_down.ch_out], [64, 64, 64])
        self.de_3 = DecAggregator([self.en_1_down.ch_out, self.en_2_down.ch_out], [64, 64])
        self.de_2 = DecAggregator([self.en_1_down.ch_out], [64])
        self.de_1 = DecAggregator([], [])

        self.head_for_classification = HeadForClassification(self.en_5_down.ch_out, self.en_5_down.ch_out)
        self.head_for_supervision_1 = HeadForSupervision(self.en_1_down.ch_out * 5, self.num_classes, self.resized_size)
        self.head_for_supervision_2 = HeadForSupervision(self.en_1_down.ch_out * 5, self.num_classes, self.resized_size)
        self.head_for_supervision_3 = HeadForSupervision(self.en_1_down.ch_out * 5, self.num_classes, self.resized_size)
        self.head_for_supervision_4 = HeadForSupervision(self.en_1_down.ch_out * 5, self.num_classes, self.resized_size)
        self.head_for_supervision_5 = HeadForSupervision(self.en_5_down.ch_out, self.num_classes,
                                                         self.resized_size)  # different ch_in alone

    def forward(self, inputs):
        # encoder
        x = self.en_1(inputs)
        x1_en = self.en_1_down(x)
        print(f"\t{x1_en.shape = }")
        x = self.en_2(x1_en)
        x2_en = self.en_2_down(x)
        print(f"\t{x2_en.shape = }")
        x = self.en_3(x2_en)
        x3_en = self.en_3_down(x)
        print(f"\t{x3_en.shape = }")
        x = self.en_4(x3_en)
        x4_en = self.en_4_down(x)
        print(f"\t{x4_en.shape = }")
        x = self.en_5(x4_en)
        x5_en = self.en_5_down(x)
        print(f"\t{x5_en.shape = }")
        print("\tencoder done")

        # decoder
        x4_de = self.de_4([x1_en, x2_en, x3_en], x4_en, [x5_en])
        print(f"\tdecoder x4_de done. {x4_de.shape = }")
        x3_de = self.de_3([x1_en, x2_en], x3_en, [x5_en, x4_en])
        print(f"\tdecoder x3_de done. {x3_de.shape = }")
        x2_de = self.de_2([x1_en], x2_en, [x5_en, x4_en, x3_en])
        print(f"\tdecoder x2_de done. {x2_de.shape = }")
        x1_de = self.de_1([], x1_en, [x5_en, x4_en, x3_en, x2_en])
        print(f"\tdecoder x1_de done. {x1_de.shape = }")

        # head for full-scale deep supervision
        #         y_hat_5 = self.head_for_classification(x5_en)
        y_hat_5 = self.head_for_supervision_5(x5_en)
        print(f"\t\t{y_hat_5.shape = }")
        y_hat_4 = self.head_for_supervision_4(x4_de)
        print(f"\t\t{y_hat_4.shape = }")
        y_hat_3 = self.head_for_supervision_3(x3_de)
        print(f"\t\t{y_hat_3.shape = }")
        y_hat_2 = self.head_for_supervision_2(x2_de)
        print(f"\t\t{y_hat_2.shape = }")
        y_hat_1 = self.head_for_supervision_1(x1_de)
        print(f"\t\t{y_hat_1.shape = }")

        #         return x4_de, x3_de, x2_de, x1_de
        return y_hat_1, y_hat_2, y_hat_3, y_hat_4, y_hat_5