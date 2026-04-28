# exe_strain_multiscenario.py
# Multi-scenario CSDI execution script.
# Example:
# python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --epochs 50 --nsample 50

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import shutil
import random
from datetime import datetime

import numpy as np
import torch
import yaml

from dataset_strain_multiscenario import (
    get_dataloader,
    save_dataset_summary,
    save_scenario_csv,
)
from main_model_strain_multiscenario import CSDI_Str_MultiScenario
from utils_strain_multiscenario import train, evaluate_and_plot


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--nsample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dryrun_dataset", action="store_true", help="데이터셋 개수/구성만 확인하고 학습은 하지 않습니다.")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.nsample is not None:
        config["eval"]["nsample"] = int(args.nsample)

    seed = int(args.seed)
    set_seed(seed)

    feature_tag = config.get("run_name", os.path.splitext(os.path.basename(args.config))[0])
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = os.path.join("save", f"strain_multi_{feature_tag}_{now}")
    os.makedirs(foldername, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")

    print("\n" + "=" * 90)
    print("[RUN] CSDI strain reconstruction / MULTI-SCENARIO INPUT SWEEP")
    print(f"[RUN] device       : {device}")
    print(f"[RUN] config       : {args.config}")
    print(f"[RUN] run_name     : {feature_tag}")
    print(f"[RUN] save folder  : {foldername}")
    print(f"[RUN] epochs       : {config['train']['epochs']}")
    print(f"[RUN] nsample      : {config['eval'].get('nsample', 50)}")
    print(f"[RUN] seed         : {seed}")
    print("=" * 90 + "\n")

    shutil.copyfile(args.config, os.path.join(foldername, os.path.basename(args.config)))

    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset, info = get_dataloader(config, seed=seed)

    with open(os.path.join(foldername, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    save_dataset_summary(os.path.join(foldername, "dataset_summary.json"), train_dataset.events, info)
    save_scenario_csv(os.path.join(foldername, "train_scenarios.csv"), train_dataset.scenarios)
    save_scenario_csv(os.path.join(foldername, "valid_scenarios.csv"), valid_dataset.scenarios)
    save_scenario_csv(os.path.join(foldername, "test_scenarios.csv"), test_dataset.scenarios)
    print("[SAVE] dataset_info.json / dataset_summary.json / scenario CSV files saved")

    if args.dryrun_dataset:
        print("[DRYRUN] 데이터셋 구성 확인만 수행하고 종료합니다.")
        return

    # 결측 target 점이 실제로 있는지 dry-run 확인
    first_batch = next(iter(train_loader))
    removed = (first_batch["observed_mask"] - first_batch["gt_mask"]).sum().item()
    removed_str = (first_batch["observed_mask"][:, :, train_dataset.str_index] - first_batch["gt_mask"][:, :, train_dataset.str_index]).sum().item()
    print(f"[CHECK] first train batch removed total points : {int(removed)}")
    print(f"[CHECK] first train batch removed STr points   : {int(removed_str)}")

    target_dim = len(info["feature_names"])
    model = CSDI_Str_MultiScenario(config, device, target_dim=target_dim).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] model target_dim       : {target_dim}")
    print(f"[INFO] feature_names          : {info['feature_names']}")
    print(f"[INFO] trainable parameters   : {trainable_params:,}")
    print(f"[INFO] diffusion num_steps    : {model.num_steps}")
    print(f"[INFO] window/stride          : {config['data'].get('window_size')} / {config['data'].get('stride')}")

    model.train()
    with torch.no_grad():
        # no_grad이지만 forward 내부 noise/loss 계산 확인용. backward는 하지 않음.
        dry_loss = model(first_batch).item()
    print(f"[CHECK] dry-run train loss before epoch 1 : {dry_loss:.8f}")
    if dry_loss == 0.0:
        print("[WARN] dry-run loss가 0입니다. gt_mask 결측 생성이 비었을 가능성이 있으니 설정을 확인하세요.")

    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        valid_epoch_interval=int(config["train"].get("valid_epoch_interval", 10)),
        foldername=foldername,
    )

    nsample = int(config.get("eval", {}).get("nsample", 50))
    evaluate_and_plot(
        model,
        test_loader,
        test_dataset,
        nsample=nsample,
        foldername=foldername,
        eval_cfg=config.get("eval", {}),
    )

    print("[DONE] strain CSDI multi-scenario training/evaluation finished")


if __name__ == "__main__":
    main()
