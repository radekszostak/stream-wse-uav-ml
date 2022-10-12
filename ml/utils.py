import torch
import numpy as np
import time
import copy
from tqdm import tqdm
from collections import defaultdict
import torch.nn.functional as F
import csv
import pdb
import logging
import pandas
import os
import pickle
from statistics import mean as list_mean
import random
import io
from contextlib import redirect_stdout
from IPython.display import display, HTML
        
def printc(text, name="Collapsed text"):
    text = str(text)
    display(HTML(f"""
    <details style="font-family: var(--jp-code-font-family); font-size: var(--jp-code-font-size)">
    <summary >{name}</summary>

    <p style="white-space:pre-line;">{text}</p>
    </details>
    """))
def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
def dictarr2arrdict(dictarr):
    info_arr=[]
    keys = list(dictarr.keys())
    for i in range(len(dictarr["name"])):
        item_dict = {}
        for key in keys:
            if isinstance(dictarr[key],torch.Tensor):
                dictarr[key]=dictarr[key].cpu().numpy()
            item_dict[key] = dictarr[key][i]
        info_arr.append(item_dict)
    return info_arr

OVERAL_STD=1.1969874641502298
ORT_MU=[0.485, 0.456, 0.406]
ORT_SIGMA=[0.229, 0.224, 0.225]
ORT_MU_BW = list_mean(ORT_MU)
ORT_SIGMA_BW = list_mean(ORT_SIGMA)
def elev_standardize(elev,mean,inverse=False):
    if inverse:
        elev = elev*(2*OVERAL_STD)+mean
    else:
        elev = (elev-mean)/(2*OVERAL_STD)
    return elev
def ort_standardize(ort,inverse=False):
    if inverse:
        ort = ort*ORT_SIGMA_BW+ORT_MU_BW
    else:
        ort = (ort-ORT_MU_BW)/(ORT_SIGMA_BW)
    return ort
def std_standardize(var,inverse=False):
    if inverse:
        var = var*(2*OVERAL_STD)
    else:
        raise NotImplementedError
    return var

def calc_mse_loss(predict, target):
    loss = F.mse_loss(predict, target)
    return loss

def calc_rmse(predict, target):
    return torch.sqrt(torch.mean(torch.square(predict-target)))

def training_evaluation(model, name, dataloader, neptune_run, device, dropout_averaging):
    model.eval()
    if dropout_averaging:
        enable_dropout(model)
        n_averaging = 20
    else:
        n_averaging = 1
    with torch.no_grad():
        averaging_eval = []
        for i in range(n_averaging):
            epoch_eval_loss = []
            predict_eval = []
            target_eval = []
            rmse_eval = []
            for data, target, info in dataloader:
                data, target, info = data.to(device), target.to(device), dictarr2arrdict(info)
                predict = model(data)
                predict_val = torch.unsqueeze(predict[:, 0],dim=1)
                loss = calc_mse_loss(predict_val, target).to(device)
                epoch_eval_loss.append(loss.item())
                predict_eval.append(predict_val.detach().cpu())
                target_eval.append(target.detach().cpu())
            epoch_eval_loss = np.mean(epoch_eval_loss)
            averaging_eval.append(torch.squeeze(torch.cat(predict_eval, dim=0)))
        std_eval, predict_eval = torch.std_mean(torch.stack(averaging_eval), dim=0)
        target_eval = torch.squeeze(torch.cat(target_eval, dim=0))
        rmse_eval = std_standardize(calc_rmse(predict_eval, target_eval), inverse=True)
        std_eval = std_standardize(std_eval, inverse=True)
        
        if neptune_run:
            neptune_run[f"{name}/rmse"].log(rmse_eval)
            neptune_run[f"{name}/std"].log(std_eval)
            #neptune_run[f"{name}/averaging"].log(averaging_eval)
            #neptune_run[f"{name}/predict"].log(predict_eval)
            #neptune_run[f"{name}/target"].log(target_eval)
            #neptune_run[f"{name}/rasidual"].log(std_standardize(predict_eval-target_eval,inverse=True))
        print(f"[{name}] loss: {epoch_eval_loss}, rmse: {rmse_eval}")
        return rmse_eval
    
def train(model, train_dataloader, valid_dataloader, test_dataloader, optimizer, scheduler, device, num_epochs, 
                    es_patience, es_min_delta, dropout_averaging, output_dir,
                    neptune_run=None):
    state_dict_path = f"{output_dir}/state_dict.pth"
    try:
        if es_patience >= 0:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_rmse = float('inf')
            es_no_improvement = 0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs}. LR: {scheduler.get_last_lr()[0]}')
            # train
            model.train()
            
            epoch_train_loss = []
            predict_train = []
            target_train = []
            rmse_train = []

            for data, target, info in tqdm(train_dataloader):
                data, target, info = data.to(device), target.to(device), dictarr2arrdict(info)
                optimizer.zero_grad()
                predict = model(data)
                predict_val = torch.unsqueeze(predict[:, 0],dim=1)
                loss = calc_mse_loss(predict_val, target).to(device)
                loss.backward()
                epoch_train_loss.append(loss.item())
                optimizer.step()
                target_train.append(target.detach().cpu())
                predict_train.append(predict_val.detach().cpu())
            epoch_train_loss = np.mean(epoch_train_loss)
            target_train = torch.squeeze(torch.cat(target_train, dim=0))
            predict_train = torch.squeeze(torch.cat(predict_train, dim=0))
            rmse_train = std_standardize(calc_rmse(predict_train, target_train), inverse=True)
            if neptune_run:
                neptune_run[f"train/rmse"].log(rmse_train)
            print(f"[train] loss: {epoch_train_loss}, rmse: {rmse_train}")
            
            # valid
            if valid_dataloader:
                rmse_valid = training_evaluation(model=model,name="valid",dataloader=valid_dataloader,neptune_run=neptune_run, device=device, dropout_averaging=dropout_averaging)
            # test
            if test_dataloader:
                rmse_test = training_evaluation(model=model,name="test",dataloader=test_dataloader,neptune_run=neptune_run, device=device, dropout_averaging=dropout_averaging)

            scheduler.step()
            if es_patience >= 0:
                es_delta = best_rmse - rmse_valid
                if es_delta > es_min_delta:
                    es_no_improvement = 0
                    print(f"Valid RMSE improved by {es_delta}. Saving best model.")
                    best_rmse = rmse_valid
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts,state_dict_path)
                else:
                    es_no_improvement += 1
                    print(f"No RMSE improvement since {es_no_improvement}/{es_patience} epochs.")
                if es_no_improvement > es_patience:
                    break
            else:
                torch.save(model.state_dict(),state_dict_path)

    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    except BaseException as err:
        #import pdb; pdb.set_trace()
        #print(f"Unexpected {err}=, {type(err)=}")
        logging.exception("message")
        raise
    finally:
        if es_patience >= 0:
            print('Best rmse_valid: {:4f}'.format(best_rmse))
            model.load_state_dict(best_model_wts)
        if type(optimizer).__name__ == "SWA":
            optimizer.swap_swa_sgd()
            # valid
            rmse_valid = training_evaluation(model=model,name="final_valid",dataloader=valid_dataloader,neptune_run=neptune_run, device=device, dropout_averaging=dropout_averaging)
            print(f"After SWA averaging validation RMSE: {rmse_valid}")
            # test
            rmse_test = training_evaluation(model=model,name="final_test",dataloader=test_dataloader,neptune_run=neptune_run, device=device, dropout_averaging=dropout_averaging)
            print(f"After SWA averaging test RMSE: {rmse_test}")
            torch.save(model.state_dict(),state_dict_path)
    return model
    
# load weights
def predict(model, dataloader, averaging, device, output_name, dropout_averaging, output_dir, neptune_run):

      
    
    model.eval()
    if dropout_averaging:
        print(f"Dropout averaging...")
        enable_dropout(model)
        n_averaging = 100
    else:
        n_averaging = 1
    averaging_val_predictions = []
    for i in range(n_averaging):
        predict_val_test = []
        with torch.no_grad():
            for data, target, info in dataloader:
                data, target = data.to(device), target.to(device)
                predict = model(data)
                predict_val = predict[:, 0]
                predict_val_test.append(predict_val.detach())
        predict_val_test = torch.squeeze(torch.cat(predict_val_test, dim=0))
        averaging_val_predictions.append(predict_val_test)
    targets = []
    names = []
    dsm_means = []
    dsm_ranges = []
    infos = []
    for data, target, info in dataloader:
        targets.append(target.detach())
        names.extend(info['name'])
        dsm_means.append(info['mean'].detach())
        dsm_ranges.append(info['max'].detach()-info['min'].detach())
        infos.extend(dictarr2arrdict(info))
    dsm_means = torch.squeeze(torch.cat(dsm_means, dim=0))
    dsm_ranges = torch.squeeze(torch.cat(dsm_ranges, dim=0))
    targets = elev_standardize(torch.squeeze(torch.cat(targets, dim=0)), dsm_means, inverse=True)
    averaging_val_predictions = torch.vstack(averaging_val_predictions).detach().cpu()
    val_means = elev_standardize(torch.mean(averaging_val_predictions,axis=0), dsm_means, inverse=True)

    
    out_dict = {"predict": pandas.DataFrame(elev_standardize(averaging_val_predictions,dsm_means,inverse=True).numpy()), "info": pandas.DataFrame.from_dict(infos)}
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    output_dict_path = f"{output_dir}/predictions/{output_name}.pickle"
    with open(output_dict_path, 'wb') as f:
        pickle.dump(out_dict,f)

    RMSE = calc_rmse(val_means, targets).item()
    print(f"{output_name} RMSE: {RMSE}")
    if neptune_run:
        neptune_run[f"predict/{output_name.split('_')[0]}/RMSE"].log(RMSE)