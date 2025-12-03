import sys
sys.path.append('..')
from utils import *

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args

        super(OthelloNNet, self).__init__()
        
        # 1. 卷积层 (Feature Extraction)
        # padding=1 确保 19x19 进去还是 19x19 出来
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # 2. 全连接层 (Policy & Value Heads)
        # 尺寸计算：64 * 19 * 19 = 23104
        self.flat_size = args.num_channels * self.board_x * self.board_y
        
        self.fc1 = nn.Linear(self.flat_size, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)  # Policy Head
        self.fc4 = nn.Linear(512, 1)                 # Value Head

    def forward(self, s):
        # s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x 19 x 19
        
        # 经过4层卷积，尺寸保持不变
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x 64 x 19 x 19
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x 64 x 19 x 19
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x 64 x 19 x 19
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x 64 x 19 x 19
        
        # 关键修改：展平 (Flatten)
        # 以前这里写死了 -4，现在直接用我们在 __init__ 里算好的 flat_size
        s = s.view(-1, self.flat_size)                               # batch_size x 23104

        # 全连接层
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        # 输出
        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)