import torch
import os
import numpy as np
import nibabel as nib
from MRINet import MRINet
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

test_dir = r'G:\MIP_MRI_Image_Analysis\MNI\Test'
target_labels = []
predicted_labels = []
model_file = r'G:\MIP_MRI_Image_Analysis\Eval_Files\model_9.pth'

###### Load model #########
model = MRINet()
model.load_state_dict(torch.load(model_file))
model.eval()
correct = 0
total = 0
####### Load files and predict ########
for alzheimer_class in os.listdir(test_dir):
    if alzheimer_class == "AD_test_set":
        label = 0
    elif alzheimer_class == "MCI_test_set":
        label = 1
    elif alzheimer_class == "CN_test_set":
        label = 2
    else:
        continue

    current_dir = os.path.join(test_dir,alzheimer_class)

    for image in os.listdir(current_dir):
        target_labels.append(label)
        total += 1
        nii_image = nib.load(os.path.join(current_dir, image))
        img_numpy_array = np.array(np.nan_to_num(nii_image.get_fdata()))
        img_numpy_array = img_numpy_array[None]
        img_numpy_array = torch.FloatTensor(img_numpy_array)
        img_numpy_array = img_numpy_array.view(1, img_numpy_array.size(0), img_numpy_array.size(1), img_numpy_array.size(2),img_numpy_array.size(3))
        img_numpy_array = Variable(img_numpy_array)
        model_output = model(img_numpy_array)
        model_output = model_output.detach().numpy()
        predicted_class = np.argmax(model_output)
        if predicted_class == label:
            correct += 1

        predicted_labels.append(predicted_class)

target_labels = np.array(target_labels)
predicted_labels = np.array(predicted_labels)
model_confusion_matrix = confusion_matrix(target_labels,predicted_labels)
modelFolder = '/home/group3/Models_Karthik'
np.save(os.path.join(modelFolder,'confusion_matrix'),model_confusion_matrix)
print("Model accuracy for all three classes = {}/{} and accuracy = {:.2f}%".format(correct,total,(correct/total)*100))
print("Confusion matrix is",model_confusion_matrix)
