import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sn
import pandas as pd

eval_dir = r'G:\MIP_MRI_Image_Analysis\Eval_Files\No_Dropout_2_Class'
accuracy = np.load(os.path.join(eval_dir,'accuracy.npy'),allow_pickle=True)
loss = np.load(os.path.join(eval_dir,'loss.npy'),allow_pickle=True )
model_confusion_matrix = np.load(os.path.join(eval_dir,'confusion_matrix.npy'),allow_pickle=True)
epochs = np.arange(20)

# plt.figure(1)
# plt.plot(epochs,loss)
# plt.title('Training Loss per epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
# plt.figure(2)
# plt.plot(epochs, accuracy)
# plt.title('Accuracy per epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.show()
fig,(ax1,ax2) = plt.subplots(2,1)

ax1.plot(epochs,loss)
ax1.set_title('Training Loss per epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(epochs,accuracy)
ax2.set_title('Accuracy per epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')

plt.show()


df_cm = pd.DataFrame(model_confusion_matrix, range(2), range(2))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

plt.show()