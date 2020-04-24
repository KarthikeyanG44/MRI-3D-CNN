from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
import utils
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import multiprocessing
import json
from MRINet import MRINet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import nibabel as nib


def occlusion(model, image_tensor, target_class=None, size=5, stride=5, occlusion_value=0, apply_softmax=True,
              three_d=True, resize=True, cuda=True, verbose=True):
    """
    Perform occlusion (Zeiler & Fergus 2014) to determine the relevance of each image pixel
    for the classification decision. Return a relevance heatmap over the input image.

    Note: The current implementation can only handle 2D and 3D images.
    It usually infers the correct image dimensions, otherwise they can be set via the `three_d` parameter.

    Args:
        model (torch.nn.Module): The pytorch model. Should be set to eval mode.
        image_tensor (torch.Tensor or numpy.ndarray): The image to run through the `model` (channels first!).
        target_class (int): The target output class for which to produce the heatmap.
                      If `None` (default), use the most likely class from the `model`s output.
        size (int): The size of the occlusion patch.
        stride (int): The stride with which to move the occlusion patch across the image.
        occlusion_value (int): The value of the occlusion patch.
        apply_softmax (boolean): Whether to apply the softmax function to the output. Useful for models that are trained
                                 with `torch.nn.CrossEntropyLoss` and do not apply softmax themselves.
        three_d (boolean): Whether the image is 3 dimensional (e.g. MRI scans).
                           If `None` (default), infer from the shape of `image_tensor`.
        resize (boolean): The output from the occlusion method is usually smaller than the original `image_tensor`.
                          If `True` (default), the output will be resized to fit the original shape (without interpolation).
        cuda (boolean): Whether to run the computation on a cuda device.
        verbose (boolean): Whether to display additional output during the computation.

    Returns:
        A numpy array of the same shape as image_tensor, indicating the relevance of each image pixel.
    """

    # TODO: Try to make this better, i.e. generalize the method to any kind of input.
    if three_d is None:
        three_d = (len(image_tensor.shape) == 4)  # guess if input image is 3D

    image_tensor = torch.Tensor(image_tensor)  # convert numpy or list to tensor
    if cuda:
        image_tensor = image_tensor.cuda()
    image_tensor = Variable(image_tensor, requires_grad=False)
    output = model(image_tensor).cpu()
    if apply_softmax:
        output = F.softmax(output)

    output_class = output.max(1)[1].data.numpy()[0]
    if verbose: print('Image was classified as', output_class, 'with probability', output.max(1)[0].data[0])
    if target_class is None:
        target_class = output_class
    unoccluded_prob = output.data[0, target_class]

    width = image_tensor.shape[1]
    height = image_tensor.shape[2]

    xs = range(0, width, stride)
    ys = range(0, height, stride)

    # TODO: Maybe use torch tensor here.
    if three_d:
        depth = image_tensor.shape[3]
        zs = range(0, depth, stride)
        relevance_map = np.zeros((len(xs), len(ys), len(zs)))
    else:
        relevance_map = np.zeros((len(xs), len(ys)))

    if verbose:
        xs = tqdm_notebook(xs, desc='x')
        ys = tqdm_notebook(ys, desc='y', leave=False)
        if three_d:
            zs = tqdm_notebook(zs, desc='z', leave=False)

    image_tensor_occluded = image_tensor.clone()  # TODO: Check how long this takes.

    if cuda:
        image_tensor_occluded = image_tensor_occluded.cuda()

    for i_x, x in enumerate(xs):
        x_from = max(x - int(size / 2), 0)
        x_to = min(x + int(size / 2), width)

        for i_y, y in enumerate(ys):
            y_from = max(y - int(size / 2), 0)
            y_to = min(y + int(size / 2), height)

            if three_d:
                for i_z, z in enumerate(zs):
                    z_from = max(z - int(size / 2), 0)
                    z_to = min(z + int(size / 2), depth)

                    # if verbose: print('Occluding from x={} to x={} and y={} to y={} and z={} to z={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))

                    image_tensor_occluded.copy_(image_tensor)
                    image_tensor_occluded[:, x_from:x_to, y_from:y_to, z_from:z_to] = occlusion_value

                    # TODO: Maybe run this batched.
                    output = model(Variable(image_tensor_occluded, requires_grad=False))
                    if apply_softmax:
                        output = F.softmax(output)

                    occluded_prob = output.data[0, target_class]
                    relevance_map[i_x, i_y, i_z] = unoccluded_prob - occluded_prob

            else:
                # if verbose: print('Occluding from x={} to x={} and y={} to y={}'.format(x_from, x_to, y_from, y_to, z_from, z_to))
                image_tensor_occluded.copy_(image_tensor)
                image_tensor_occluded[:, x_from:x_to, y_from:y_to] = occlusion_value

                # TODO: Maybe run this batched.
                output = model(Variable(image_tensor_occluded, requires_grad=False))
                if apply_softmax:
                    output = F.softmax(output)

                occluded_prob = output.data[0, target_class]
                relevance_map[i_x, i_y] = unoccluded_prob - occluded_prob

    relevance_map = np.maximum(relevance_map, 0)
    if resize:
        relevance_map = utils.resize_image(relevance_map, image_tensor.shape[2:])

    return relevance_map


model_file = r'G:\MIP_MRI_Image_Analysis\Eval_Files\No_Dropout_2_Class\model_19.pth'
model = MRINet()
model.load_state_dict(torch.load(model_file))
model.eval()
model.cuda()
image_path = r'G:\MIP_MRI_Image_Analysis\Interpretation_Files\CN_Test.nii'
nii_image = nib.load(image_path)
mri_image = np.array(np.nan_to_num(nii_image.get_fdata()))
img_numpy_array = mri_image[None]
img_numpy_array = torch.FloatTensor(img_numpy_array)
img_numpy_array = img_numpy_array.view(1, img_numpy_array.size(0), img_numpy_array.size(1), img_numpy_array.size(2),img_numpy_array.size(3))
image_class = 1
# AD - 0
# MCI - 1
# CN - 2
interpret_map = occlusion(model,img_numpy_array,image_class,cuda=True,verbose=True)
utils.plot_slices(mri_image,overlay=interpret_map)