def main(config: dict, wandb_run=None):
    """
    Args:
        config: dictionary with the following keys:
            "method": str, "direct", "mask" or "fusion"
            "architecture": str, parameter for smp.create_model function
            "encoder": str, parameter for smp.create_model function
            "encoder_weights": str, parameter for smp.create_model function
            "batch_size": int, batch size
            "learning_rate": float, learning rate
            "k_fold_subset": int or str, subset for validation. Available options: 0, 1, 2, 3, 4, "GRO21", "RYB21", "GRO20", "RYB20", "AMO18"
        wandb_run: Weights and Biases run object
    """
    import torch
    import numpy as np
    import pandas as pd
    import segmentation_models_pytorch as smp
    import os
    
    os.makedirs("predictions", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Detected GPUs: {torch.cuda.device_count()}")


    class Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset_dir, names=None, augment=False):

            self.augment = augment
            self.dataset_dir = dataset_dir

            #load dataset.csv
            columns = ["name","wse","dsm_mean","dsm_std","dsm_min","dsm_max","chain","lat","lon","subset"]
            self.dataset_df = pd.read_csv(f"{dataset_dir}/dataset.csv", names=columns, header=0)
            
            #filter by names
            if names is not None:
                self.dataset_df = self.dataset_df[self.dataset_df["name"].isin(names)]

            #create augmentation columns
            if augment:
                flip_x_col = []; flip_y_col = []; rot_90_col = []
                for fx in [False, True]:
                    for fy in [False, True]:
                        for r in [0, 1, 2, 3]:
                            flip_x_col.append(fx); flip_y_col.append(fy); rot_90_col.append(r)
                augment_combinations = pd.DataFrame({'flip_x': flip_x_col, 'flip_y': flip_y_col, 'rot_90': rot_90_col})
            else:
                augment_combinations = pd.DataFrame({'flip_x': [False], 'flip_y': [False], 'rot_90': [0]})
            self.dataset_df = self.dataset_df.merge(augment_combinations, how='cross')

        def __len__(self):
            return len(self.dataset_df)

        def __getitem__(self, i):
            #load dsm and ort
            dsm = np.load(f"{self.dataset_dir}/dsm/{self.dataset_df['name'].iloc[i]}")
            ort = np.load(f"{self.dataset_dir}/ort/{self.dataset_df['name'].iloc[i]}")

            #augment
            if self.augment:
                if self.dataset_df["flip_x"].iloc[i]:
                    dsm = np.flip(dsm, axis=0)
                    ort = np.flip(ort, axis=0)
                if self.dataset_df["flip_y"].iloc[i]:
                    dsm = np.flip(dsm, axis=1)
                    ort = np.flip(ort, axis=1)
                if self.dataset_df["rot_90"].iloc[i] != 0:
                    dsm = np.rot90(dsm, k=self.dataset_df["rot_90"].iloc[i])
                    ort = np.rot90(ort, k=self.dataset_df["rot_90"].iloc[i])

            #return dictionary
            sample_dict = self.dataset_df.iloc[i][["wse","dsm_mean","dsm_std","name","chain","subset"]].to_dict()
            sample_dict["dsm"] = dsm
            sample_dict["ort"] = ort
            
            return sample_dict

    def collate_fn(data):
        #dictionary of dictionaries to dictionary of lists
        batch_dict = {}
        for key in data[0].keys():
            batch_dict[key] = [d[key] for d in data]
        #convert to torch tensors
        for key in batch_dict.keys():
            batch_dict[key] = torch.tensor(np.stack(batch_dict[key]), dtype=torch.float32).unsqueeze(1) if key in ["dsm","ort","wse","dsm_mean","dsm_std"] else batch_dict[key]
        return batch_dict

    #standardization
    DSM_OVERAL_STD = 1.1969874641502298
    ORT_MU=[0.485, 0.456, 0.406]
    ORT_SIGMA=[0.229, 0.224, 0.225]
    ORT_MU_BW = np.mean(ORT_MU)
    ORT_SIGMA_BW = np.mean(ORT_SIGMA)
    def wse_standardize(wse,dsm_mean,inverse=False):
        if inverse:
            return wse*(DSM_OVERAL_STD)+dsm_mean
        else:
            return (wse-dsm_mean)/(DSM_OVERAL_STD)
    def dsm_standardize(dsm,dsm_mean, inverse=False):
        if inverse:
            return dsm*(DSM_OVERAL_STD)+dsm_mean[:,None,None]
        else:
            return (dsm-dsm_mean[:,None,None])/(DSM_OVERAL_STD)
    def ort_standardize(ort,inverse=False):
        if inverse:
            return ort*ORT_SIGMA_BW+ORT_MU_BW
        else:
            return (ort-ORT_MU_BW)/ORT_SIGMA_BW

    dataset_df = pd.read_csv(f"dataset/dataset.csv", header=0)
    all_names = dataset_df["File name"].to_list()
    if type(config["k_fold_subset"]) == int:
        assert config["k_fold_subset"] in [0, 1, 2, 3, 4]
        valid_names = [all_names[i] for i in range(len(all_names)) if i % 5 == config["k_fold_subset"]]
    elif type(config["k_fold_subset"]) == str:
        assert config["k_fold_subset"] in ['GRO21', 'RYB21', 'GRO20', 'RYB20', 'AMO18']
        valid_names = dataset_df[dataset_df["Subset"]==config["k_fold_subset"]]["File name"].to_list()
    train_names = list(set(all_names) - set(valid_names))

    train_dataset = Dataset(dataset_dir="dataset", names=train_names, augment=True)
    valid_dataset = Dataset(dataset_dir="dataset", names=valid_names, augment=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    
    model = smp.create_model(arch=config["architecture"], encoder_name=config["encoder"], encoder_weights=config["encoder_weights"], in_channels=2, classes=1, activation="sigmoid", aux_params={"classes": 1, "activation": None}).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    calc_loss = torch.nn.MSELoss()

    torch.cuda.empty_cache()
    best_valid_rmse = np.inf
    best_valid_details = None
    es_patience = 20
    es_counter = 0
    for epoch in range(10000):
        print(f"Epoch: {epoch}")
        train_loss = []
        train_details = []
        model.train()
        for batch in train_dataloader:
            dsm, ort, wse = dsm_standardize(batch["dsm"],batch["dsm_mean"]).to(device), ort_standardize(batch["ort"]).to(device), wse_standardize(batch["wse"], batch["dsm_mean"]).to(device)
            x = torch.cat([dsm, ort], dim=1)
            optimizer.zero_grad()
            mask_hat, offset_hat = model(x)
            if config["method"] == "direct":
                wse_hat = offset_hat
            elif config["method"] == "mask":
                wse_hat = torch.sum(dsm*mask_hat, dim=(2,3))/torch.sum(mask_hat, dim=(2,3))
            elif config["method"] == "fusion":
                wse_hat = torch.sum(dsm*mask_hat, dim=(2,3))/torch.sum(mask_hat, dim=(2,3)) + offset_hat
            loss = calc_loss(wse_hat, wse)
            loss.backward()
            optimizer.step()
            wse_hat = wse_standardize(wse_hat.detach().cpu(), batch["dsm_mean"], inverse=True)
            train_details.append({"wse": batch["wse"].squeeze().tolist(), "wse_hat": wse_hat.squeeze().tolist(), "name": batch["name"], "chain": batch["chain"], "subset": batch["subset"]})
            train_loss.append(loss.item())
            
        train_loss = np.mean(train_loss)
        train_details = pd.DataFrame(pd.concat([pd.DataFrame.from_dict(d) for d in train_details]))
        train_rmse = np.sqrt(np.mean((train_details["wse"]-train_details["wse_hat"])**2))
        
        with torch.no_grad():
            model.eval()
            valid_details = []
            for batch in valid_dataloader:
                dsm, ort, wse = dsm_standardize(batch["dsm"],batch["dsm_mean"]), ort_standardize(batch["ort"]), wse_standardize(batch["wse"], batch["dsm_mean"])
                dsm, ort, wse = dsm.to(device), ort.to(device), wse.to(device)
                x = torch.cat([dsm, ort], dim=1)
                mask_hat, offset_hat = model(x)
                if config["method"] == "direct":
                    wse_hat = offset_hat
                elif config["method"] == "mask":
                    wse_hat = torch.sum(dsm*mask_hat, dim=(2,3))/torch.sum(mask_hat, dim=(2,3))
                elif config["method"] == "fusion":
                    wse_hat = torch.sum(dsm*mask_hat, dim=(2,3))/torch.sum(mask_hat, dim=(2,3)) + offset_hat
                wse_hat = wse_standardize(wse_hat.detach().cpu(), batch["dsm_mean"], inverse=True)
                valid_details.append({"wse": batch["wse"].squeeze().tolist(), "wse_hat": wse_hat.squeeze().tolist(), "name": batch["name"], "chain": batch["chain"], "subset": batch["subset"], "method": config["method"]})
            valid_details = pd.DataFrame(pd.concat([pd.DataFrame.from_dict(d) for d in valid_details]))
            valid_rmse = np.sqrt(np.mean((valid_details["wse"]-valid_details["wse_hat"])**2))
        if wandb_run:
            wandb_run.log({"train_rmse": train_rmse, "valid_rmse": valid_rmse})    
        
        #early stopping
        if valid_rmse < best_valid_rmse:
            print(f"Valid RMSE improved.")
            best_valid_rmse = valid_rmse
            best_valid_details = valid_details.copy()
            es_counter = 0
    
            torch.save(model.state_dict(), f"checkpoints/{config['output_name']}.pth")
        else:
            es_counter += 1
            print(f"Valid RMSE not improved since {es_counter} epochs.")
        if es_counter == es_patience:
            break
    best_valid_details.to_csv(f"predictions/{config['output_name']}.csv")