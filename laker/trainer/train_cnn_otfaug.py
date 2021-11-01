"""
Cnn-Transformer Training script
"""
import sys
import argparse
import importlib
import os
import os.path
import math

import numpy as np
#torch related
import torch
import torch.optim as optim
from torch._six import inf
import torch.nn as nn
import torchvision.transforms as transforms


MASTER_NODE = 0

def run_one_epoch(epoch, model, log_f,
                  args, training):
    """
    Run one epoch of training

    Args:
        epoch (int): zero based epoch index
        model (torch.nn.module): model
        log_f (file): logging file
        args : arguments from outer
        training (bool): training or validation
    """
    log_f.write('===> Epoch {} <===\n'.format(epoch))
    total_num_batches = args.num_epochs * args.num_batches_per_epoch
    num_batches_processed = epoch * args.num_batches_per_epoch
    lr = args.initial_lr / args.seq_len * math.exp(num_batches_processed * \
                                    math.log(args.final_lr /\
                                             args.initial_lr) /\
                                    total_num_batches)
    log_f.write('===Using Learning Rate {}===\n'.format(lr))
    #optimizer = optim.SGD(model.parameters(), lr,
    #                      momentum=args.momentum,
    #                      nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr)
    transform = transforms.ConvertImageDtype(torch.float)

    if training:
        model.train()
    else:
        model.eval()
        optimizer = None

    for num_done, data_cpu in \
        enumerate(args.dataloader(args.data_dir, args)):
        print("num_done %d\n" %(num_done))
        if training:
            optimizer.zero_grad()

        if data_cpu is not None:
            #forward bsz, frame, C, H, W
            data_batch = data_cpu.cuda(args.local_rank)
            data_batch = transform(data_batch) 
            outputs = model.forward(data_batch)
            loss = nn.MSELoss()(data_batch, outputs)
            if num_done % 2 == 0:
                print('Epoch :', epoch,'|','train_loss:%.4f'%loss.data)
        else: #empty batch
            loss = torch.FloatTensor([0.0])

        if training:
            if data_cpu is not None:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.grad_clip,
                                                   norm_type=inf)
                optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Imag Encoder training')

    parser.add_argument('nnet_proto', type=str,
                        help='pytorch NN proto definition filename')
    parser.add_argument('log', type=str,
                        help='log file for the job')
    parser.add_argument('output_dir', type=str,
                        help='path to save the final model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='data directory')
    parser.add_argument('--init_model', type=str, default=None,
                        help='initial model')
    parser.add_argument('--input_H', type=int, default=640,
                        help='initial Height of Images')
    parser.add_argument('--input_W', type=int, default=480,
                        help='initial Weight of Images')
    parser.add_argument('--cnn_layers', type=int, default=5,
                        help='num of cnn layers')
    parser.add_argument('--transformer_layers', type=int, default=1,
                        help='num of transformer layers')
    parser.add_argument('--stride', type=int, default=2,
                        help='stride used in cnn layers')

    parser.add_argument('--grad_clip', type=float, default=-1.0,
                        help='gradient clipping threshold')

    parser.add_argument('--initial_lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--final_lr', type=float, default=1.0,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')
    parser.add_argument('--num_epochs', type=int, default=15,
                        help='number of epochs for training')
    parser.add_argument('--num_batches_per_epoch', type=int, default=1000,
                        help='number of batches per work per epoch')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    parser.add_argument('--loader', choices=['otf_imag'],
                        default='otf_imag',
                        help='loaders')
    parser.add_argument('--seed', type=int, default=777,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')

    #BMUF related
    parser.add_argument('--local_rank', type=int,
                        help='local process ID for parallel training')

    args, unk = parser.parse_known_args()

    # import loader
    #loader_module = importlib.import_module('loader.' + args.loader \
    #                                         + '_loader')
    loader_module = importlib.import_module('loader.otf_imag_loader') 
    loader_module.register(parser)
    args = parser.parse_args()
    args.dataloader = loader_module.dataloader
    log_f = open(args.log, 'w')
    # set cuda device
    assert args.cuda
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.local_rank)

    #manual seed for reproducibility
    nnet_module = importlib.import_module("model."+args.nnet_proto)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if args.init_model is None:
        def weights_init(m):
            if type(m) == nn.Linear:
               nn.init.xavier_normal_(m.weight.data)
            if type(m) == nn.Conv2d:
               nn.init.xavier_normal_(m.weight.data)
        model = nnet_module.Net(args)
        model.apply(weights_init)
    else:
        model = torch.load(args.init_model,
                           map_location=lambda storage, loc: storage)

    model.cuda(args.local_rank)

    num_param = 0
    for param in model.parameters():
        num_param += param.numel()
    #print model proto
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('model proto: {}\n'
                'model size: {} M\n'
                .format(args.nnet_proto, num_param/1000/1000))
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.write('*'*60+'\n')
    log_f.flush()

    if torch.cuda.is_available():
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
            model.cuda()
        else:
            print('WARNING: You have a CUDA device, '
                  'so you should probably run with --cuda')
    #start training with run_one_epoch, params:
    #(model, args, bmuf_trainer, train or valid)
    for epoch in range(0, args.num_epochs):
        run_one_epoch(epoch, model, log_f,
                                   args, True)
        #save current model
        current_model = '{}/model.epoch.{}.{}'.format(args.output_dir,
                                                      epoch, args.local_rank)
        with open(current_model, 'wb') as f:
            torch.save(model, f)

    log_f.write('Training Finished')
