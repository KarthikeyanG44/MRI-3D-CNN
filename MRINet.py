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
        self.dense_1 = nn.Linear(32*8*11*10, 64)
        self.dense_2 = nn.Linear(64, 16)
        self.dense_3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.Conv_1_bn(self.Conv_1(x)))
        x = F.max_pool3d(x, 2)
        x = self.relu(self.Conv_2_bn(self.Conv_2(x)))
        x = F.max_pool3d(x, 3)
        x = self.relu(self.Conv_3_bn(self.Conv_3(x)))
        x = F.max_pool3d(x, 2)
        x = x.view(-1,32*8*11*10)
        x = self.relu(self.dense_1(x))
        x = self.relu(self.dense_2(x))
        x = self.dense_3(x)
        return F.softmax(x, dim=1)
