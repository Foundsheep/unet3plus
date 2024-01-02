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


class DecConv(nn.Module):
    def __init__(self, enc_list, dec_list):
        super().__init__()
        down_layers = []
        enc_same = enc_list[-1]

        for idx, enc in enumerate(enc_list):
            if idx == len(enc_list) - 1:
                continue
            stri = int(enc_same.ch_out / enc.ch_out)
            down_layer = DownSampling(ch_in=enc.ch_out, ch_out=64, stri=stri, to_concat=True)
            down_layers.append(down_layer)

        # for the same-shaped tensor
        self.conv = nn.Conv2d(enc_same.ch_out, 64, 3, 1, 1)
        down_layers.append(self.conv)
        self.down_layers = nn.ModuleList(down_layers)

        # for tensors from decoder
        up_layers = []
        for dec in dec_list:
            factor = int(enc_same.h_out / dec.h_out)
            up_layer = UpSampling(ch_in=dec.ch_out, ch_out=64, factor=factor, to_concat=True)
            up_layers.append(up_layer)
        self.up_layers = nn.ModuleList(up_layers)

    def forward(self, enc_list, dec_list):
        x_list = []
        for enc, down_layer in zip(enc_list, self.down_layers):
            x_list.append(down_layer(enc))

        for dec, up_layer in zip(dec_list, self.up_layers):
            x_list.append(up_layer(dec))

        x = torch.concatenate(x_list, axis=1)
        return x