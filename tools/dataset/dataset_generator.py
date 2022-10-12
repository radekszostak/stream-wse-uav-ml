import numpy as np
import os
import cv2
import shutil
import csv
import random

######################
basename = "x_KOC1_" ##
######################
if basename == "x_KOC1_":
    raise ValueError('Basename not specified.')
os.chdir(os.path.dirname(os.path.realpath(__file__)))
img_size = (256,256)
out_dir = "ds2ndpshseout"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)

os.makedirs(out_dir)

os.makedirs(os.path.join(out_dir,"dsm"))
os.makedirs(os.path.join(out_dir,"ort"))
csv_path = os.path.join(out_dir,"dataset_.csv")
with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    for name in os.listdir("ort"):

        info = os.path.splitext(name)[0].split("_")
        i = info[0]
        level = info[1]
        chainage = info[2]
        centroid_y = info[3]
        centroid_x = info[4]
        ort_path = os.path.join("ort",name)
        dsm_path = os.path.join("dsm",name)
        
        ort = np.load(ort_path)
        dsm = np.load(dsm_path)
        ort = np.moveaxis(ort, 0, -1)
        ort = cv2.resize(ort,img_size)
        ort = np.moveaxis(ort, -1, 0)
        dsm = cv2.resize(dsm,img_size)
        new_name = basename+i
        new_ort_path = os.path.join(out_dir,"ort",new_name)
        new_dsm_path = os.path.join(out_dir,"dsm",new_name)

        random.seed(new_name)
        phase = "train" if random.random()>0.2 else "test"

        writer.writerow([new_name, level, np.mean(dsm), np.std(dsm), np.amin(dsm), np.amax(dsm), chainage, centroid_y, centroid_x, phase ])
        np.save(new_ort_path, ort)
        np.save(new_dsm_path, dsm)
