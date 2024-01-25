# About
This repostory contains code and machine learning dataset used in article ***Estimation of Small Stream Water Surface Elevation Using UAV Photogrammetry and Deep Learning*** by Radosław Szostak, Marcin Pietroń, Przemysław Wachniew, Mirosław Zimnoch and Paweł Ćwiąkała (AGH UST).

# Requirements
Code must be run in Python 3. Required libraries: *torch*, [*segmentation_models_pytorch*](https://github.com/qubvel/segmentation_models.pytorch), *numpy*, *pandas*.

# Running
See [run.py](https://github.com/radekszostak/stream-wse-uav-ml/blob/master/run.py) for example code for running the solution.

[main.py](https://github.com/radekszostak/stream-wse-uav-ml/blob/master/main.py) contains single funtion handling both training and evaluation. Function contains docstring to help understand its parameters. Function will create *predictions* and *checkpoints* folders in running directory.

# Acknowledgments
Research was partially supported by the National Science Centre, Poland, project WATERLINE (2020/02/Y/ST10/00065), under the CHISTERA IV programme of the EU Horizon 2020 (Grant no 857925) and the "Excellence Initiative - Research University" program at the AGH University of Science and Technology. The computing resources of the PL-Grid infrastructure were used in this study.
