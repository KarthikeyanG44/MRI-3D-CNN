#### Import Dependencies ###
import torch.nn as nn
import torch.nn.functional as F

####### Define the network #####
class MRINet(nn.Module):
    def __init__(self,):
        super(MRINet, self).__init__()
        self.Conv_1 = nn.Conv3d(1, 8, 3,stride=1)
        self.Conv_1_bn = nn.BatchNorm3d(8)
        self.Conv_2 = nn.Conv3d(8, 16, 3,stride=1)
        self.Conv_2_bn = nn.BatchNorm3d(16)
        self.Conv_3 = nn.Conv3d(16, 32, 3,stride=1)
        self.Conv_3_bn = nn.BatchNorm3d(32)
        self.Conv_4 = nn.Conv3d(32, 64, 3,stride=1)
        self.Conv_4_bn = nn.BatchNorm3d(64)
        # self.dropout = nn.Dropout3d(p = 0.2)
        self.dense_1 = nn.Linear(64*1*2*2,64)
        self.dense_2 = nn.Linear(64, 32)
        self.dense_3 = nn.Linear(32,16)
        self.dense_4 = nn.Linear(16,8)
        self.dense_5 = nn.Linear(8,3)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = F.max_pool3d(x, 3)
        # x = self.dropout(x)
        # print("After convolution 1",x.size())
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = F.max_pool3d(x, 3)
        # x = self.dropout(x)
        # print("After convolution 2",x.size())
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = F.max_pool3d(x, 2)
        # print("After convolution 3",x.size())
        x = self.relu(self.Conv_4_bn(self.Conv_4(x)))
        x = F.max_pool3d(x,2)
        # print("After convolution 4",x.size())
        x = x.view(-1,64*1*2*2)
        x = self.relu(self.dense_1(x))
        x = self.relu(self.dense_2(x))
        x = self.relu(self.dense_3(x))
        x = self.relu(self.dense_4(x))
        x = self.dense_5(x)
        return F.log_softmax(x, dim=1)
