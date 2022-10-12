from this import d
import albumentations
import os
import numpy
import matplotlib.pyplot as plt

dsm_dir = "dataset/dsm"
ort_dir = "dataset/ort"

common_augment = albumentations.Compose(
    [
        #albumentations.HorizontalFlip(p=0.5),
        #albumentations.VerticalFlip(p=0.5),
        albumentations.RandomResizedCrop(p=1, height=256, width=256, scale=(0.4, 0.9), ratio=(0.5, 2.0)),
    ],
    additional_targets={'dsm': 'image'}
  )
ort_augment = albumentations.Compose(
  [
    #albumentations.Equalize(p=1,mode="pil"),
    albumentations.GaussNoise(p=1,var_limit=(0,0.02)),
    albumentations.MotionBlur(p=1),
    #albumentations.MotionBlur(p=0.2),
    #albumentations.RandomBrightnessContrast(p=0.2),
  ]
)
dsm_augment = albumentations.Compose(
  [
    #albumentations.Equalize(p=0.2),
    albumentations.GaussNoise(p=1,var_limit=(0,0.02)),
    #albumentations.MotionBlur(p=1),
    #albumentations.RandomBrightnessContrast(p=0.2),
  ]
)

for file in os.listdir(dsm_dir):
    dsm = numpy.load(os.path.join(dsm_dir, file))
    ort = numpy.load(os.path.join(ort_dir, file))
    #plot dsm and ort image side by side
    fig, ax = plt.subplots(2, 2, figsize=(10, 5))
    ax[0,0].imshow(dsm)
    ax[0,1].imshow(ort)
    #apply common augmentations
    augmented = common_augment(image=ort, dsm=dsm)
    ort = augmented['image']
    dsm = augmented['dsm']
    ort = ort_augment(image=ort)['image']
    dsm = dsm_augment(image=dsm)['image']
    
    #plot augmented images side by side
    #fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[1,0].imshow(dsm)
    ax[1,1].imshow(ort)
    
    plt.show()