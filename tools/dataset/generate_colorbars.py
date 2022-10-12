import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
files = ["20181121_AMO_020.npy","20201219_GRO_037.npy","20201219_RYB_020.npy","20210713_GRO_030.npy","20210713_RYB_006.npy"]
ticks = [,[212.8, 213.6, 214.4]]
file = files[4]
dsm = np.load(f"dsm/{file}")
print(file)
name = file.split(sep="_")[1]+file[2:4]
plt.imshow(dsm, cmap='viridis')
plt.axis('off')
ax=plt.gca()
for PCM in ax.get_children():
    if isinstance(PCM, mpl.cm.ScalarMappable):
        break
cb = plt.colorbar(PCM, ax=ax, orientation="horizontal", fraction=0.047)
cb.ax.tick_params(labelsize=20)
cb.outline.set_visible(False)
cb.update_ticks()
tick_locator = mpl.ticker.FixedLocator([212.8, 213.6, 214.4])
cb.locator = tick_locator
cb.update_ticks()
ax.remove()
#plt.show()
plt.savefig(f'img/dsmcb_{name}.png', transparent=True, bbox_inches='tight', pad_inches=0)
cb.remove()