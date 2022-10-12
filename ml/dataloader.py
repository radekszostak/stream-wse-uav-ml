from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
from utils import elev_standardize, ort_standardize, printc
import csv
from pathlib import Path


PERMUTATION = 16
ALL_SUBSETS = ["GRO21","RYB21","GRO20","RYB20","AMO18"]
class WseDataset(Dataset):
    def __init__(
            self, 
            csv_path,
            img_size,
            augment=False,
            normalize=True,
            names=[],
            subsets=None,
            min_range=0.,
            max_range=float("inf")
    ):
      
        assert all(item in ALL_SUBSETS for item in subsets)

        self.csv_path = csv_path
        self.subsets = subsets
        self.dir = os.path.dirname(csv_path)
        self.img_size = (img_size, img_size)
        self.augment = augment
        self.normalize = normalize
        self.dsm_dir = os.path.join(self.dir,"dsm")
        self.ort_dir = os.path.join(self.dir,"ort")
        self.min_range = min_range
        self.max_range = max_range
        self.all_info_arr = self.info_reader()
        self.info_arr = []
        self.names = []
        if names:
            for d in self.all_info_arr:
                if d["name"] in names:
                    self.names.append(d["name"])
                    self.info_arr.append(d)
        elif subsets:
            for d in self.all_info_arr:
                if d["subset"] in self.subsets:
                    rng = (d["max"]-d["min"])
                    if (rng < self.min_range) or (rng > self.max_range):
                        continue
                    self.names.append(d["name"])
                    self.info_arr.append(d)

        self.dsm_fps = [os.path.join(self.dsm_dir, name) for name in self.names]
        self.ort_fps = [os.path.join(self.ort_dir, name) for name in self.names]
        printc(f"Number of samples: {len(self.names)}.\nSample names: {self.names}", f"Subsets: {subsets}")

    def info_reader(self):
        info_arr = []
        with open(os.path.join(self.csv_path), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            first_row=True
            for row in csv_reader:
                if first_row: 
                    first_row=False
                    continue
                else:
                    info_arr.append({   "name": row[0],
                                        "wse": float(row[1]),
                                        "mean": float(row[2]),
                                        "std": float(row[3]),
                                        "min": float(row[4]),
                                        "max": float(row[5]),
                                        "chain": float(row[6]),
                                        "lat": float(row[7]),
                                        "long": float(row[8]),
                                        "subset": row[9]
                                    })
        return info_arr

    def augmentation(self, dsm, ort, y, i):
        m16 = i%16
        m4 = m16%4
        rotation = m16//4    #values from 0 to 3
        flip_x = m4 in [1,3]#
        flip_y = m4 in [2,3]# 0 - no flip, 1 - only flip x, 2 - only flip y, 3 - flip x and y.
        if rotation != 0:
            ort = np.copy(np.rot90(ort,rotation,(2,1)))
            dsm = np.copy(np.rot90(dsm,rotation,(2,1)))
        if flip_x:
            ort = np.copy(np.flip(ort,1))
            dsm = np.copy(np.flip(dsm,1))
        if flip_y:
            ort = np.copy(np.flip(ort,2))
            dsm = np.copy(np.flip(dsm,2))
        return (dsm, ort, y)
   
    def __getitem__(self, i):
        if self.augment:
            sample_i = i//16
        else:
            sample_i = i
        
        assert os.path.basename(self.dsm_fps[sample_i])==self.info_arr[sample_i]["name"]
        dsm = np.load(self.dsm_fps[sample_i])
        dsm = cv2.resize(dsm,self.img_size)
        dsm = np.expand_dims(dsm,0)
        ort = np.load(self.ort_fps[sample_i])
        ort = cv2.resize(ort,self.img_size)
        ort = np.expand_dims(ort,0)
        
        y = np.array([self.info_arr[sample_i]["wse"]]).astype(np.float32)
        if self.augment:
            dsm, ort, y = self.augmentation(dsm, ort, y, i)
        
        if self.normalize==True:
            dsm = elev_standardize(dsm, self.info_arr[sample_i]['mean'])
            ort = ort_standardize(ort)
            y = elev_standardize(y, self.info_arr[sample_i]['mean'])
        x = np.vstack((dsm, ort))
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y, self.info_arr[sample_i]


    def __len__(self):
        length = len(self.names)
        if self.augment:
            length = length*PERMUTATION
        return length