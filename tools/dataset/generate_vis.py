import os
import numpy as np
# load pickle from dsm, ort_bw and ort_color directories
#create img directory if not exists
os.makedirs('img', exist_ok=True)
files = ["20181121_AMO_020.npy","20201219_GRO_037.npy","20201219_RYB_020.npy","20210713_GRO_030.npy","20210713_RYB_006.npy"]
for file in files:
    dsm = np.load(f"dsm/{file}")
    ort_color = np.load(f"ort_color/{file}")
    ort_bw = np.load(f"ort_bw/{file}")
    print(file)
    name = file.split(sep="_")[1]+file[2:4]
    #plot the dsm, ort_bw and ort_color on the same plot
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.imshow(dsm, cmap='viridis')
    plt.axis('off')
    plt.savefig(f'img/dsm_{name}.png', transparent=True, bbox_inches='tight', pad_inches=0)
    
    
    plt.imshow(ort_bw, cmap='gray')
    plt.axis('off')
    plt.savefig(f'img/ortbw_{name}.png', transparent=True, bbox_inches='tight', pad_inches=0)
    ort_color = np.moveaxis(ort_color, 0, -1)
    plt.imshow(ort_color)
    plt.axis('off')
    plt.savefig(f'img/ortcolor_{name}.png', transparent=True, bbox_inches='tight', pad_inches=0)


