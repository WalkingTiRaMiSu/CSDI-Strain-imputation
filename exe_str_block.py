import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import datetime
import json
import yaml

from main_model import CSDI_Physio
from dataset_str_block import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="Vanilla CSDI for strain-to-strain block imputation")
parser.add_argument("--config", type=str, default="base_str_block.yaml")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=20)
parser.add_argument("--data_dir", type=str, default="./processed_str_05")

args = parser.parse_args()
print(args)

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/str_block_05_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

with open(os.path.join(foldername, "config.json"), "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    data_dir=args.data_dir,
    batch_size=config["train"]["batch_size"],
    val_ratio=0.2,
    scenario_per_window=5,
    missing_ratio=0.2,
    seed=args.seed,
)

target_dim = 1
model = CSDI_Physio(config, args.device, target_dim=target_dim).to(args.device)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=1,
    foldername=foldername,
)