import numpy as np
import matplotlib.pyplot as plt
import os

eval_dir = r'G:\MIP_MRI_Image_Analysis\Eval_Files'
accuracy = np.load(os.path.join(eval_dir,'accuracy.npy'),allow_pickle=True)
loss = np.load(os.path.join(eval_dir,'loss.npy'),allow_pickle=True )
epochs = np.arange(10)

plt.figure(1)
plt.plot(list(range(10)),loss)
plt.title('Training Loss per epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.figure(2)
plt.plot(list(range(10)), accuracy)
plt.title('Accuracy per epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()