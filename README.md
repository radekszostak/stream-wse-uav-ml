# About
This repostory contains code and machine learning dataset used in article ***Estimation of river water surface elevation using UAV photogrammetry and machine learning*** by Radosław Szostak, Marcin Pietroń, Przemysław Wachniew, Mirosław Zimnoch and Paweł Ćwiąkała (AGH UST).

# Requirements
Code must be run in Python 3. File [requirements.txt](https://github.com/radekszostak/river-wse-uav-ml/blob/master/requirements.txt) contains list of required python libraries.

# Running the training
Use [ml/main.ipynb](https://github.com/radekszostak/river-wse-uav-ml/blob/master/ml/main.ipynb) notebook to run single training and evaluation.

Use [ml/k_fold.sh](https://github.com/radekszostak/river-wse-uav-ml/blob/master/ml/k_fold.sh) script to run training and evaluation for each k_fold subset.

# Result visualisation
Plots and result tables featured in article are generated using scripts and data from [plots](https://github.com/radekszostak/river-wse-uav-ml/tree/master/plots) directory.

# Acknowledgments
Research was partially supported by the National Science Centre, Poland, project WATERLINE (2020/02/Y/ST10/00065), under the CHISTERA IV programme of the EU Horizon 2020 (Grant no 857925) and the "Excellence Initiative - Research University" program at the AGH University of Science and Technology.
