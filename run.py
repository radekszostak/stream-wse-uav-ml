from main import main
import wandb
import traceback

learning_rate = 0.00001
encoder_weights = "imagenet"
encoder = "vgg19"
architecture = "PSPNet"
method = "mask"
batch_size = 4
k_fold_subset = "AMO18"
output_name = "output"

config = {"method": method,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "encoder": encoder,
        "architecture": architecture,
        "k_fold_subset": k_fold_subset,
        "encoder_weights": encoder_weights,
        "output_name": output_name}

wandb_run = None # configure a wandb run here
try:
    main(config=config, wandb_run=wandb_run)
except Exception as e:
    print(traceback.format_exc())
    wandb_run.finish(1)
else:
    wandb_run.finish(0)