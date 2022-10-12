import sys
sys.path.append('ml')
from torch.utils.data import DataLoader
from dataloader import WseDataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


dir = "dataset/train"
test_set = WseDataset(csv_path="dataset/dataset.csv", phase="test", img_size=256, augment=False)
dataloader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

for x, y, info in dataloader:
    print(f"x: {x.size()}, {torch.min(x)}, {torch.max(x)}")
    print(f"x[0]: {x[:,0].size()}, {torch.min(x[:,0])}, {torch.max(x[:,0])}")
    print(f"x[1:4]: {x[:,1:4].size()}, {torch.min(x[:,1:4])}, {torch.max(x[:,1:4])}")
    print(f"y: {y.size()}, {torch.min(y)}, {torch.max(y)}")
    imgs = x[:,0].unsqueeze(1)
    grid_img = torchvision.utils.make_grid(imgs, nrow=8)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()
