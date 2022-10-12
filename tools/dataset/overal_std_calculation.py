import os
import numpy as np
import math
os.chdir(os.path.dirname(os.path.realpath(__file__)))
n=0
std_sum=0


mean = 181.1746458466702

for file in os.listdir("dsm"):
    dsm = np.load(os.path.join("dsm",file))
    mean=dsm.mean()
    for row in dsm:
        for pixel in row:
            n+=1
            std_sum+=(pixel-mean)**2
std=math.sqrt(std_sum/n)
print(std)