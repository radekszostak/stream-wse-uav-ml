import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
def show_image(arr):
    plt.imshow(arr,cmap='gray')
    plt.show()
in_dir = "dataset/ort"
out_dir = "dataset/ort_gray"
os.makedirs(out_dir,exist_ok=True)
for file in os.listdir(in_dir):
    ort = np.load(f"{in_dir}/{file}").astype(np.float32)/255
    #->(C,H,W)
    ort = np.moveaxis(ort, 0, -1)
    #->(H,W,C)
    #ort = cv2.resize(ort,self.img_size)
    #import pdb; pdb.set_trace()
    ort = cv2.cvtColor(ort, cv2.COLOR_RGB2GRAY)
    #ort = np.expand_dims(ort,-1)
    #ort = np.moveaxis(ort, -1, 0)
    np.save(f"{out_dir}/{file}",ort)