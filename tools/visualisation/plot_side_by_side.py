import numpy as np
import matplotlib.pyplot as plt


# helper function to plot x, ground truth and predict images in grid
def plot_side_by_side(*arg):

    y_size = arg[0].shape[0]
    x_size = len(arg)
    fig, axs = plt.subplots(y_size, x_size)
    if y_size==1:
        axs = np.expand_dims(axs,0)
    for y in range(y_size):
        first_dem = True
        for x in range(x_size):
            img = arg[x][y]
            if len(img.shape)==3:
                img = np.moveaxis(img, 0, -1)
                axs[y, x].imshow(img)
            elif len(img.shape)==2:
                if first_dem:
                    min_val = np.amin(img)
                    max_val = np.amax(img)
                    first_dem = False
                axs[y, x].imshow(img, vmin = min_val, vmax = max_val)
            axs[y, x].axis('off')
            #x_ort = x[i,1:4]#.permute(1, 2, 0)
            #axs[i, 0].imshow(x_ort)

            #x_dem = x[i,0]
            #min_val = torch.min(x_dem)
            #max_val = torch.max(x_dem)
            #axs[i, 1].imshow(x_dem, vmin = min_val, vmax = max_val)
            #axs[i, 2].imshow(np.squeeze(y_dem_gt[i]), vmin = min_val, vmax = max_val)
            #if y_dem_pr is not None:
            #    axs[i, 3].imshow(denormalize(np.squeeze(y_dem_pr[i]),"dem"), vmin = min_val, vmax = max_val)
    axs = np.squeeze(axs)
    
    plt.show()
    return fig

