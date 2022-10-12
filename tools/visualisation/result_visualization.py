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


plt.rc('font', family='serif')
dir = "dataset"
x_dsm_dir = os.path.join(dir,"x_dsm")
x_ort_dir = os.path.join(dir,"x_ort")
y_dsm_dir_gt = os.path.join(dir,"y_dsm")
y_dsm_dir_pr = "predictions"
names = list(os.listdir(y_dsm_dir_pr))
names.reverse()

x_dsm_fps = [os.path.join(x_dsm_dir, name) for name in names]
x_ort_fps = [os.path.join(x_ort_dir, name) for name in names]
y_dsm_fps_gt = [os.path.join(y_dsm_dir_gt, name) for name in names]
y_dsm_fps_pr = [os.path.join(y_dsm_dir_pr, name) for name in names]
img_size = (256,256)

for i in range(len(x_ort_fps)):
    create_dir("tmp/result_img")
    x_dsm = np.load(x_dsm_fps[i])
    x_dsm = cv2.resize(x_dsm,img_size)


    x_ort = np.load(x_ort_fps[i]).astype(np.float32)
    x_ort = np.moveaxis(x_ort, 0, -1)
    x_ort = cv2.resize(x_ort,img_size)
    x_ort = np.moveaxis(x_ort, -1, 0)
    x_ort = x_ort/255

    y_dsm_gt = np.load(y_dsm_fps_gt[i])
    y_dsm_gt = cv2.resize(y_dsm_gt,img_size)

    y_dsm_pr = np.load(y_dsm_fps_pr[i])
    y_dsm_pr = cv2.resize(y_dsm_pr,img_size)
    #x = np.vstack((x_dsm, x_ort))
    #y = y_dsm_gt
    #image = image_transform(image)
    #mask = mask.astype(np.float32)
    print(x_ort_fps[i])

    fig, axs = plt.subplots(2, 2)
    min_val = np.amin(x_dsm)
    max_val = np.amax(x_dsm)
    x_ort = np.moveaxis(x_ort, 0, -1)
    axs[0, 0].imshow(x_ort)
    axs[0, 0].get_xaxis().set_ticks([])
    axs[0, 0].get_yaxis().set_ticks([])
    #axs[0, 0].title.set_text("Orthophoto")
    axs[0, 1].imshow(x_dsm, vmin = min_val, vmax = max_val, cmap="viridis")
    axs[0, 1].get_xaxis().set_ticks([])
    axs[0, 1].get_yaxis().set_ticks([])
    #axs[0, 1].title.set_text("Raw photogrammetric DSM")
    axs[1, 0].imshow(y_dsm_gt, vmin = min_val, vmax = max_val, cmap="viridis")
    axs[1, 0].get_xaxis().set_ticks([])
    axs[1, 0].get_yaxis().set_ticks([])
    #axs[1, 0].title.set_text("Ground truth DSM")
    axs[1, 1].imshow(y_dsm_pr, vmin = min_val, vmax = max_val, cmap="viridis")
    axs[1, 1].get_xaxis().set_ticks([])
    axs[1, 1].get_yaxis().set_ticks([])
    #axs[1, 1].title.set_text("DSM corrected using ML")
    xy = None
    xy = plt.ginput(2)
    
    colors = [["black","tab:green"], ["tab:orange","tab:blue"]]
    styles = [["-",":"], ["-.", "--"]]
    if xy:
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        for i in range(2):
            for j in range(2):
                #axs[i,j].axis('off')
                axs[i,j].plot(x,y,"-", color="white",linewidth=4)
                axs[i,j].plot(x,y,styles[i][j], color=colors[i][j],linewidth=2)
        length = int(np.hypot(x[1]-x[0], y[1]-y[0]))
        x, y = np.linspace(x[0], x[1], length), np.linspace(y[0], y[1], length)
        fig.savefig(r"tmp\result_img\r_spat.png", dpi=500)
        plt.show()
        
        x_data = np.array([ element*10/256 for element in range(length) ])
        x_dsm_zi =          x_dsm[y.astype(int), x.astype(int)]
        y_dsm_gt_zi =       y_dsm_gt[y.astype(int), x.astype(int)]
        y_dsm_pr_zi =       y_dsm_pr[y.astype(int), x.astype(int)]
        plt.plot(x_data, x_dsm_zi, styles[0][1], color=colors[0][1], linewidth=2, label="Raw photogrammetric DSM")
        plt.plot(x_data, y_dsm_gt_zi, styles[1][0], color=colors[1][0], linewidth=2, label="Ground truth DSM")
        plt.plot(x_data, y_dsm_pr_zi, styles[1][1], color=colors[1][1], linewidth=2, label="DSM corrected using ML")
        plt.margins(x=0)
        plt.ylabel("Elevation (m MSL)")
        plt.xlabel("Distance (m)")
        plt.legend(loc="upper right")
        plt.savefig(r"tmp\result_img\r_ax.png", dpi=500)
        plt.show()
        
    #plot_side_by_side(x_ort,x_dsm,y_dsm_gt,y_dsm_pr)
