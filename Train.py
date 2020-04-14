import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import os
from MRINet import MRINet
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F


##### Define Data Loader #####
class MRI_Loader(Dataset):
    def __init__(self):
        root_dir = r'G:\MIP_MRI_Image_Analysis\Data'
        sub_dir = os.listdir(root_dir)
        self.data = []
        self.labels = []
        for directory in sub_dir:
            new_path = os.path.join(root_dir, directory)
            if directory == "AD":
                label = 0
            elif directory == "MCI":
                label = 1
            else:
                label = 2
            for _, _, files in os.walk(new_path):
                for file in files:
                    nii_image = nib.load(os.path.join(new_path, file))
                    img_numpy_array = nii_image.get_fdata()
                    self.data.append(img_numpy_array)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        return image,label



if __name__ == "__main__":
    model = MRINet()
    root_dir = r'G:\MIP_MRI_Image_Analysis\Data'
    train_loader = DataLoader(MRI_Loader,batch_size = 64)
    if torch.cuda.is_available():
        use_gpu = True
        print("Using GPU")
    else:
        use_gpu = False
        print("Using CPU")

    lossDict = dict()
    accuracyDict = dict()
    trainLoss = []
    accur = []


    #### GPU STUFF ###
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.7)
    #### Code for training #####
    def train(epoch):
        model.train()


        correct = 0
        training_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            if use_gpu:
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            max_index = output.max(dim=1)[1]
            correct += (max_index == target).sum()
            training_loss += loss.item()

        print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            training_loss / len(train_loader.dataset), correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

        lossDict[epoch] = training_loss / len(train_loader.dataset)
        accuracyDict[epoch] = 100. * correct / len(train_loader.dataset)
        trainLoss.append(lossDict[epoch])
        accur.append(accuracyDict[epoch])

    for epoch in range(10):
        train(epoch)