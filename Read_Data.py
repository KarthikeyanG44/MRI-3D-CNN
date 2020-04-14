import nilearn as nil
from nilearn import plotting
import os
import numpy as np
path = r'E:\Gatech Spring 2020\MIP\MRI Project\Example Images\AD'
images = os.listdir(path)

# plotting.plot_anat(os.path.join(path,images[-1]))
analyze_img = nil.image.load_img(os.path.join(path,images[-3]))
# filtered_image = nil.image.smooth_img(analyze_img,fwhm= None)
plotting.plot_anat(analyze_img)