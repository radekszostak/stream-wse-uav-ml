import os
import numpy as np
from numpy.core.numeric import NaN
import cv2
import matplotlib.pyplot as plt
import csv

dir = "arc_script/dataset"
x_dem_dir = os.path.join(dir,"x_dem")
x_ort_dir = os.path.join(dir,"x_ort")
y_dem_dir_gt = os.path.join(dir,"y_dem")
names = os.listdir(x_ort_dir)#["20.npy", "37.npy", "5.npy", "52.npy", "56.npy", "83.npy"]

x_dem_fps = [os.path.join(x_dem_dir, name) for name in names]
x_ort_fps = [os.path.join(x_ort_dir, name) for name in names]
y_dem_fps_gt = [os.path.join(y_dem_dir_gt, name) for name in names]
img_size = (256,256)

saveData = []
for i in range(len(x_ort_fps)):
    x_dem = np.load(x_dem_fps[i])
    x_dem = cv2.resize(x_dem,img_size)
    offset = np.mean(x_dem)
    x_dem = x_dem - offset
    assert x_dem.shape == img_size
    np.save(x_dem_fps[i],x_dem)
    x_ort = np.load(x_ort_fps[i]).astype(np.float32)
    x_ort = np.moveaxis(x_ort, 0, -1)
    x_ort = cv2.resize(x_ort,img_size)
    x_ort = np.moveaxis(x_ort, -1, 0)
    x_ort = x_ort/255

    y_dem_gt = np.load(y_dem_fps_gt[i])
    y_dem_gt = cv2.resize(y_dem_gt,img_size)
    y_dem_gt = y_dem_gt - offset
    assert y_dem_gt.shape == img_size
    np.save(y_dem_fps_gt[i],y_dem_gt)
    print(x_ort_fps[i])

    fig, axs = plt.subplots(2)
    min_val = np.amin(x_dem)
    max_val = np.amax(x_dem)
    x_ort = np.moveaxis(x_ort, 0, -1)
    axs[0].imshow(x_ort)
    axs[0].axis('off')
    axs[0].title.set_text("Orthophoto")
    axs[1].imshow(y_dem_gt, vmin = min_val, vmax = max_val, cmap="jet")
    axs[1].axis('off')
    axs[1].title.set_text("Ground truth DEM")
    yx = None
    n = 3
    yx = plt.ginput(n)
    if yx:
        mean = np.mean([y_dem_gt[int(x),int(y)] for (y,x) in yx ])
        water = 1
    else:
        mean = "None"
        water = 0
    saveRow = [os.path.basename(x_ort_fps[i]), mean, offset]
    saveData.append(saveRow)
    print(saveRow)

    #np.mean([y_dem_gt[int(x),int(y)] for (x,y) in yx ])
    print("-")
with open(os.path.join(dir,'levels.csv'), mode='w') as levels_file:
    levels_writer = csv.writer(levels_file, delimiter=',', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    levels_writer.writerows(saveData)
