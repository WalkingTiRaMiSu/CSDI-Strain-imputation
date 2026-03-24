import argparse
import torch
import datetime
import json
import yaml
import os
import pandas as pd

from main_model import CSDI_Custom
from dataset_custom import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI Custom Multi-Sensor")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--device", default="cuda")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["test_missing_ratio"] = 0.0
config["train"]["batch_size"] = 1

print(json.dumps(config, indent=4))

# ts 제외한 센서 개수 자동 계산
df_tmp = pd.read_csv("./custom_data/original.csv")
target_dim = df_tmp.shape[1] - 1

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/custom_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(batch_size=1)

model = CSDI_Custom(config, args.device, target_dim=target_dim).to(args.device)

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

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)