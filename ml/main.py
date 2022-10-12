# -*- coding: utf-8 -*-
import os
import sys

#set workdir
#os.chdir("/content/drive/MyDrive/DEM-waterlevel/ml/")

#imports
import numpy as np
import torch
from swa import SWA
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import WseDataset
from utils import train, predict
from models.vgg import Vgg
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
import pickle
import math
import matplotlib.pyplot as plt
#training parameters in neptune format
PARAMS = {
  "img_size": 256,
  "model": "vgg",#"vgg"
  "lr_base": 0.000001,#0.00001,
  "lr_max": 0.00001,#0.00001,
  "swa_start": 6,
  "swa_freq": 2,
  "swa_lr": 0.000002,
  "batch_size": 32,
  'epochs': 50,
  'task': "all",#"all", "train", "predict"
  'neptune': True,
  'es_patience': 10,#15, # early stopping (es), negative - early stopping disabled
  'es_min_delta': 0.001, # early stopping (es)
  'drop_rate': 0.5, # dropout rate
  'k_fold_valid_subset': None,# "RYB20", "RYB21", "GRO20", "GRO21", "AMO18"
  'k_fold_test_subset': None,# None
  'dropout_averaging': True,
  'overtrain': False,
  'min_range': 0.,#4.,
  'max_range': 4.5,#float('inf'),
  'pretrained_weights': None,#None,# None,"state_dict.pth",
  'optimizer': "adam"#"adam"/"swa"/"sgd"
}
if len(sys.argv)>1:
  PARAMS['k_fold_valid_subset']=sys.argv[1]
  PARAMS['k_fold_test_subset']=sys.argv[2]
assert PARAMS["task"] in ["predict", "train", "all"]
print(PARAMS)

# neptune initialization
if PARAMS["neptune"]:
  import configparser
  config = configparser.ConfigParser()
  assert os.path.exists("./ml/config.cfg")
  config.read("./ml/config.cfg")
  import neptune.new as neptune
  neptune_run = neptune.init(project=config["neptune"]["project"],
              api_token=config["neptune"]["token"],
              source_files=['ml/main.py', f"ml/models/{PARAMS['model']}.py", 'ml/utils.py', 'ml/dataloader.py']
              )
  neptune_run["parameters"] = PARAMS
else:
  neptune_run=None

# device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")
if str(device) == "cuda":
  for i in range(torch.cuda.device_count()):
    print(f"\tcuda:{i}: {torch.cuda.get_device_name(i)}")

# model loading
if PARAMS["model"]=="vgg":
  model = Vgg(input_size=PARAMS["img_size"], drop_rate=PARAMS["drop_rate"]).to(device)
if PARAMS["pretrained_weights"]:
  model.load_state_dict(torch.load(PARAMS["pretrained_weights"]))

print(model)
# dataset configuration
dataset_dir = os.path.normpath("dataset")
csv_path = os.path.join(dataset_dir,"dataset.csv")

# train
if PARAMS["task"] in ["train", "all"]:
  if PARAMS["overtrain"]:
    train_dataset = valid_dataset = WseDataset(csv_path=csv_path, names=["20210713_GRO_033.npy", "20210713_RYB_023.npy", "20181121_AMO_027.npy"], phase = 'train', img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=None)
  else:
    train_dataset = WseDataset(csv_path=csv_path, phase="train", img_size=PARAMS['img_size'], augment=True, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])
    valid_dataset = WseDataset(csv_path=csv_path, phase="valid", img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])
    test_dataset = WseDataset(csv_path=csv_path, phase="test", img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])

  train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=True)
  valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['batch_size'], shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['batch_size'], shuffle=False)
  if PARAMS["optimizer"] == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr_base"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=PARAMS["lr_base"], max_lr=PARAMS["lr_max"],step_size_up=2,mode="triangular",cycle_momentum=False)
  elif PARAMS["optimizer"] == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=PARAMS["learning_rate"])
  elif PARAMS["optimizer"] == "swa":
    #base_optimizer = torch.optim.SGD(model.parameters(), lr=PARAMS["learning_rate"])
    base_optimizer = torch.optim.Adam(model.parameters(), lr=PARAMS["lr_base"])
    optimizer = SWA(base_optimizer, swa_start=PARAMS["swa_start"], swa_freq=PARAMS["swa_freq"], swa_lr=PARAMS["lr_max"])
    
    
  model = train(model=model,
                train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                num_epochs=PARAMS['epochs'],
                es_patience=PARAMS['es_patience'],
                es_min_delta=PARAMS['es_min_delta'],
                dropout_averaging=PARAMS["dropout_averaging"],
                neptune_run=neptune_run
               )

if PARAMS["task"] in ["predict", "all"]:
  train_dataset = WseDataset(csv_path=csv_path, phase="train", img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])
  train_dataloader = DataLoader(train_dataset, batch_size=PARAMS['batch_size'], shuffle=False)
  valid_dataset = WseDataset(csv_path=csv_path, phase="valid", img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])
  valid_dataloader = DataLoader(valid_dataset, batch_size=PARAMS['batch_size'], shuffle=False)
  test_dataset = WseDataset(csv_path=csv_path, phase="test", img_size=PARAMS['img_size'], augment=False, k_fold_valid_subset=PARAMS["k_fold_valid_subset"], k_fold_test_subset=PARAMS["k_fold_test_subset"], min_range = PARAMS["min_range"], max_range = PARAMS["max_range"])
  test_dataloader = DataLoader(test_dataset, batch_size=PARAMS['batch_size'], shuffle=False)

  predict(model, dataloader=train_dataloader, averaging=PARAMS['dropout_averaging'], device=device, output_name=f"train_{PARAMS['k_fold_valid_subset']}_{PARAMS['k_fold_test_subset']}", dropout_averaging=PARAMS["dropout_averaging"], neptune_run=neptune_run)
  predict(model, dataloader=valid_dataloader, averaging=PARAMS['dropout_averaging'], device=device, output_name=f"valid_{PARAMS['k_fold_valid_subset']}_{PARAMS['k_fold_test_subset']}", dropout_averaging=PARAMS["dropout_averaging"], neptune_run=neptune_run)
  predict(model, dataloader=test_dataloader, averaging=PARAMS['dropout_averaging'], device=device, output_name=f"test_{PARAMS['k_fold_valid_subset']}_{PARAMS['k_fold_test_subset']}", dropout_averaging=PARAMS["dropout_averaging"], neptune_run=neptune_run)
  result_dict_path = f"predictions/test_{PARAMS['k_fold_valid_subset']}_{PARAMS['k_fold_test_subset']}.pickle"
  neptune_run[f"predict/test/result_dict"].upload(result_dict_path)
  with open(result_dict_path, 'rb') as f:
    result_dict = pickle.load(f)
    gt_y = result_dict["info"]["wse"].to_numpy()
    mean_y = result_dict["predict"].mean(axis=0).to_numpy()
    std_y = result_dict["predict"].std(axis=0).to_numpy()
    fit_x = result_dict["info"]["chain"].to_numpy()
    gt_reg = sm.OLS(gt_y, sm.add_constant(fit_x)).fit()
    pr_reg = sm.OLS(mean_y, sm.add_constant(fit_x)).fit()
    pr_st, pr_data, pr_ss2 = summary_table(pr_reg, alpha=0.05)
    gt_st, gt_data, gt_ss2 = summary_table(gt_reg, alpha=0.05)
    pred_y = pr_data[:,2]
    gt_pred_y = gt_data[:,2]
    #predict_ci_low, predict_ci_upp = pr_data[:,6:8].T
    b, a = pr_reg.params
    b_std, a_std = pr_reg.bse
    predict_low = (a-a_std)*fit_x + (b-b_std)
    predict_upp = (a+a_std)*fit_x + (b+b_std)
    errors = predict_upp - pred_y
    regression_rmse = np.sqrt(((pred_y-result_dict["info"]["wse"])**2).mean())
    mean_error = (pred_y-predict_low).mean()
    print(f"Test regression RMSE: {regression_rmse}")
    neptune_run[f"predict/test/regression_RMSE"] = regression_rmse
    print(f"Test mean error: {mean_error}")
    neptune_run[f"predict/test/regression_error"] = mean_error
    slopes = {"slope_mm_gt": (gt_pred_y[-1]-gt_pred_y[0])/(fit_x[-1]-fit_x[0]),
              "slope_deg_gt": math.degrees(math.atan(gt_reg.params[1])),
              "slope_mm_pr": (pred_y[-1]-pred_y[0])/(fit_x[-1]-fit_x[0]),
              "slope_deg_pr": math.degrees(math.atan(pr_reg.params[1]))}
    neptune_run[f"predict/test/slopes"] = slopes
    neptune_run[f"predict/test/slope_diff_deg"] = slopes["slope_deg_gt"]-slopes["slope_deg_pr"]
    plt.plot(result_dict["info"]["chain"],gt_pred_y, "-", label="Ground truth linear", color="yellow")
    plt.plot(result_dict["info"]["chain"],result_dict["info"]["wse"], "-", label="Ground truth", color="green")
    plt.plot(result_dict["info"]["chain"],result_dict["predict"].mean(axis=0), "x", label="CNN output", color="blue")
    plt.plot(result_dict["info"]["chain"],pred_y, "-", label="CNN output regression", color="blue")
    plt.plot(result_dict["info"]["chain"],predict_low, "--", label="Error Up", color="blue")
    plt.plot(result_dict["info"]["chain"],predict_upp, "--", label="Error Down", color="blue")
    neptune_run[f"predict/test/figure"].upload(neptune.types.File.as_image(plt.gcf()))
  if PARAMS["neptune"]:
    neptune_run.stop()

