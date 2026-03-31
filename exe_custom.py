import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import torch
import datetime
import json
import yaml
import pandas as pd

from dataset_custom import get_dataloader
from utils import train, evaluate
from main_model_boundary import CSDI_Custom_Boundary


def get_feature_columns(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if str(c).startswith("ch_")]
    if len(feature_cols) == 0:
        raise ValueError(f"ch_ 로 시작하는 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
    return feature_cols


parser = argparse.ArgumentParser(description="CSDI Custom Multi-Channel + Boundary Anchoring")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument("--device", default="cuda")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=20)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = False
config["model"]["test_missing_ratio"] = 0.0

print(json.dumps(config, indent=4))

df_tmp = pd.read_csv("./custom_data/original.csv")
feature_cols = get_feature_columns(df_tmp)
target_dim = len(feature_cols)

print("사용 feature:", feature_cols)
print("target_dim:", target_dim)

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/custom_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

train_loader, valid_loader, test_loader = get_dataloader(
    batch_size=config["train"]["batch_size"],
    window_len=128,
    stride=16,
    train_missing_ratio=0.2,
    seed=42,
)

model = CSDI_Custom_Boundary(config, args.device, target_dim=target_dim).to(args.device)

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