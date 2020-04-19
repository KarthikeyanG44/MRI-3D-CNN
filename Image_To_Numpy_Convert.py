import numpy as np
import nibabel as nib
import os

if __name__=="__main__":
    root_dir = r'G:\MIP_MRI_Image_Analysis\Data_Original'
    sub_dir = os.listdir(root_dir)
    target_dir = r'G:\MIP_MRI_Image_Analysis\Data_Numpy'
    for classes in sub_dir:
        for images in os.listdir(os.path.join(root_dir,classes)):
            numpy_dir = os.path.join(target_dir, classes,images.split('.')[0])
            nii_image = nib.load(os.path.join(root_dir, classes,images))
            img_numpy_array = np.array(nii_image.dataobj)
            np.save(numpy_dir,img_numpy_array)
        print("Class",classes,"finished")

    print("All classes export to Numpy finished!!")