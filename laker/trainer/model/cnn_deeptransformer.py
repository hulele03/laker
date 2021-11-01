"""
this module implements TDNN transformer
for transducer encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trainer.model.modules.transformer import TransformerEncoderLayer

class Net(nn.Module):
    """
    Class implements TDNN transformer used for transducer
    encoder in "Minimum Bayes Risk Training of RNN-Transducer
    for End-to-End Speech Recognition", InterSpeech2020

    Args:
        cnn_layers (int): number of CNN layers, note that we insert
                          1 transformer layer after each 3 TDNN layers
        bn_dim (int): deprecated

    """
    def __init__(self, opt, bn_dim=0):
        super(Net, self).__init__()
        self.input_H = opt.input_H
        self.input_W = opt.input_W
        self.cnn_layers = cnn_layers =  opt.cnn_layers
        self.transformer_layers = opt.transformer_layers 
        self.stride = stride = opt.stride
        filter_size = 3
        self.filter_size = filter_size
        #hidden layers
        assert self.cnn_layers > (3 + 1)
        self.hidden_conv = nn.ModuleList(
            [nn.Conv2d(in_channels=3, out_channels=3,
                       kernel_size=(filter_size, filter_size),
                       stride=(stride, stride),
                       dilation=(1, 1)) for i in range(cnn_layers)])

        self.hidden_bn = nn.ModuleList(
            [nn.BatchNorm2d(3) for i in range(cnn_layers)])
        
        self.hidden_H = self.input_H 
        self.hidden_W = self.input_W
        for i in range(cnn_layers):
            self.hidden_H = np.floor((self.hidden_H - (filter_size - 1) - 1) // stride + 1) 
            self.hidden_H = int(self.hidden_H)
            self.hidden_W = np.floor((self.hidden_W - (filter_size - 1) - 1) // stride + 1) 
            self.hidden_W = int(self.hidden_W)
        ### after convolution, the size of the H*W
        cnn_nhid = self.hidden_H * self.hidden_W
        self.cnn_nhid_16 = cnn_nhid_16 = cnn_nhid - cnn_nhid % 16

        ## To make the hidden size is divisible by 16
        self.fc_mid = nn.Linear(cnn_nhid, cnn_nhid_16) 
        self.bn_final = nn.BatchNorm1d(3 * cnn_nhid_16)
        num_heads = 8
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(3 * cnn_nhid_16, num_heads,
                                     3 * cnn_nhid_16 * 4, 0.2,
                                     max_relative_positions=0) for i in range(self.transformer_layers)])
        ## the decoder layers
        self.fc_out1 = nn.Linear(cnn_nhid_16, 2048)
        self.fc_out2 = nn.Linear(2048, self.input_H * self.input_W)


    def forward(self, x, frame_offset=0):
        #x: batch, frame, C, H, W (dim)
        bsz = x.size()[0]
        frame = x.size()[1] 
        x_seq = list()
        for l, (conv, bn) in enumerate(zip(self.hidden_conv, self.hidden_bn)):
            # Calculate the Convolution frame by frame
            for t in range(frame): 
                xt = x[:,t,:,:,:]
                xt = conv(xt)
                xt = bn(F.relu(xt)).unsqueeze(1)
                x_seq.append(xt)
            # batch, frame, C, H, W
            x = torch.cat(x_seq, 1)
            x_seq = list()  
        x = x.view(bsz, frame, 3, -1)
        x = self.fc_mid(x).view(bsz, frame, -1).contiguous()
        for i in range(self.transformer_layers):
            x = self.transformer[i](x, mask=None)
        x = x.view(-1, 3 * self.cnn_nhid_16)
        x = self.bn_final(x)
        x = x.view(bsz, frame, 3, self.cnn_nhid_16).contiguous()
        x = nn.Tanh()(self.fc_out1(x))
        x = nn.Sigmoid()(self.fc_out2(x))
        x = x.view(bsz, frame, 3, self.input_H, self.input_W).contiguous() 
        return x
