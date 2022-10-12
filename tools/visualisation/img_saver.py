

"""
import matplotlib.pyplot as plt
class LineDrawer(object):
    lines = []
    def draw_line(self):
        ax = plt.gca()
        xy = plt.ginput(2)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        line = plt.plot(x,y)
        ax.figure.canvas.draw()

        self.lines.append(line)
ld = LineDrawer()
ld.draw_line()
"""

import os
import numpy as np
import torch
from helper import plot_side_by_side, create_dir
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

create_dir("mpl_img")

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

dir = "dataset"
x_dsm_dir = os.path.join(dir,"x_dsm")
x_ort_dir = os.path.join(dir,"x_ort")
y_dsm_dir_gt = os.path.join(dir,"y_dsm")
#y_dsm_dir_pr = "predictions"
names = ["20201219_KOC2_026.npy"]#list(os.listdir(y_dsm_dir_pr))

x_dsm_fps = [os.path.join(x_dsm_dir, name) for name in names]
x_ort_fps = [os.path.join(x_ort_dir, name) for name in names]
y_dsm_fps_gt = [os.path.join(y_dsm_dir_gt, name) for name in names]
#y_dsm_fps_pr = [os.path.join(y_dsm_dir_pr, name) for name in names]
img_size = (256,256)

for i in range(len(x_ort_fps)):
    x_dsm = np.load(x_dsm_fps[i])
    x_dsm = cv2.resize(x_dsm,img_size)


    x_ort = np.load(x_ort_fps[i]).astype(np.float32)
    x_ort = np.moveaxis(x_ort, 0, -1)
    x_ort = cv2.resize(x_ort,img_size)
    x_ort = np.moveaxis(x_ort, -1, 0)
    x_ort = x_ort/255

    y_dsm = np.load(y_dsm_fps_gt[i])
    y_dsm = cv2.resize(y_dsm,img_size)

    #y_dsm_pr = np.load(y_dsm_fps_pr[i])
    #y_dsm_pr = cv2.resize(y_dsm_pr,img_size)
    #x = np.vstack((x_dsm, x_ort))
    #y = y_dsm_gt
    #image = image_transform(image)
    #mask = mask.astype(np.float32)
    print(x_ort_fps[i])

    fig, axs = plt.subplots(1, 4)
    min_val = np.amin(x_dsm)
    max_val = np.amax(x_dsm)
    print(f"{min_val}, {max_val}")
    #min_val = math.floor(min_val/0.5)*0.5
    #max_val = math.ceil(max_val/0.5)*0.5
    print(f"{min_val}, {max_val}")
    x_ort = np.moveaxis(x_ort, 0, -1)
    cmap = mpl.cm.viridis

    
    fig = plt.figure(frameon=False, figsize=(2,2))
    ax = plt.Axes(fig, [0.05, 0.05, 0.9, 0.9])

    #ax.set_axis_off()
    fig.add_axes(ax)
    #ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.imshow(x_ort)
    fig.savefig(r"mpl_img\d_x_ort.png", dpi=200)
    ax.imshow(x_dsm, vmin = min_val, vmax = max_val, cmap=cmap)
    fig.savefig(r"mpl_img\d_x_dsm.png", dpi=200)
    ax.imshow(y_dsm, vmin = min_val, vmax = max_val, cmap=cmap)
    fig.savefig(r"mpl_img\d_y_dsm.png", dpi=200)
    fig.clear()
    fig = plt.figure(frameon=False, figsize=(1,2))
    ax = plt.Axes(fig, [0.1, 0.05, 0.2, 0.9])
    ax.set_axis_on()
    fig.add_axes(ax)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(ax, norm=norm, cmap=cmap, orientation='vertical', label='Elevation (m MSL)')
    #ax.set_axis_on()
    fig.savefig(r"mpl_img\d_cb.png", dpi=200)
    
    """
    axs[0].imshow(x_ort)
    axs[0].axis('off')
    axs[0].title.set_text("Orthophoto")
    axs[1].imshow(x_dsm, vmin = min_val, vmax = max_val, cmap=cmap)
    axs[1].axis('off')
    axs[1].title.set_text("Input dsm")
    axs[2].imshow(y_dsm, vmin = min_val, vmax = max_val, cmap=cmap)
    axs[2].axis('off')
    axs[2].title.set_text("Ground truth dsm")
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(axs[3], norm=norm, cmap=cmap, orientation='vertical', label='m.a.s.l.')
    #axs[1, 1].imshow(y_dsm_pr, vmin = min_val, vmax = max_val, cmap="jet")
    #axs[1, 1].axis('off')
    #axs[1, 1].title.set_text("Deep learning corrected dsm")
    fig.show()
    input('press <ENTER> to continue')
    """