# exe_strain_global.py
# Global STr block missing 전용 실행 파일.
# 기존 baseline / plus 파일은 건드리지 않고, 이 파일과 base_strain_global.yaml만 새로 실행한다.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import time
from datetime import datetime

import yaml
import torch
import matplotlib.pyplot as plt

from main_model_strain_global import CSDI_StrainGlobal
from dataset_strain_global import (
    get_dataloader,
    save_dataset_summary,
    save_scenario_csv,
)
from utils_strain_global import train, evaluate_and_plot

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def make_save_folder(prefix="strain_global"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("save", f"{prefix}_{now}")
    os.makedirs(folder, exist_ok=True)
    return folder


def apply_cli_overrides(config, args):
    if args.epochs is not None:
        config["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["train"]["batch_size"] = int(args.batch_size)
    if args.lr is not None:
        config["train"]["lr"] = float(args.lr)
    if args.window_size is not None:
        config["data"]["window_size"] = int(args.window_size)
    if args.stride is not None:
        config["data"]["stride"] = int(args.stride)
    if args.active_ratio is not None:
        config["data"]["active_ratio"] = float(args.active_ratio)
    if args.active_min is not None:
        config["data"]["active_min"] = int(args.active_min)
    if args.active_max is not None:
        config["data"]["active_max"] = int(args.active_max)
    if args.num_steps is not None:
        config["diffusion"]["num_steps"] = int(args.num_steps)
    if args.train_jitter_repeats is not None:
        config["data"]["train_jitter_repeats"] = int(args.train_jitter_repeats)
    return config


def main():
    parser = argparse.ArgumentParser(description="CSDI strain GLOBAL block reconstruction")
    parser.add_argument("--config", type=str, default="base_strain_global.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--nsample", type=int, default=500)
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--use_best", type=int, default=1, help="1이면 validation best model로 최종 평가")

    # 빠른 실험용 override. 안 쓰면 yaml 값 사용.
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--active_ratio", type=float, default=None)
    parser.add_argument("--active_min", type=int, default=None)
    parser.add_argument("--active_max", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--train_jitter_repeats", type=int, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA를 사용할 수 없어 CPU로 실행합니다.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config = apply_cli_overrides(config, args)

    foldername = args.modelfolder if args.modelfolder else make_save_folder(prefix="strain_global")
    os.makedirs(foldername, exist_ok=True)

    config_save_path = os.path.join(foldername, "config_used.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    print("\n" + "=" * 90)
    print("[RUN] CSDI strain GLOBAL block reconstruction")
    print(f"[RUN] device       : {device}")
    print(f"[RUN] config       : {args.config}")
    print(f"[RUN] save folder  : {foldername}")
    print(f"[RUN] nsample      : {args.nsample}")
    print(f"[RUN] seed         : {args.seed}")
    print("=" * 90 + "\n")

    train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset, info = get_dataloader(
        config, seed=args.seed
    )

    # dataset/scenario 기록 저장
    save_dataset_summary(os.path.join(foldername, "dataset_summary_global.json"), test_dataset.events, info)
    with open(os.path.join(foldername, "dataset_info_global.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    save_scenario_csv(os.path.join(foldername, "train_scenarios_global.csv"), train_dataset.scenarios)
    save_scenario_csv(os.path.join(foldername, "valid_scenarios_global.csv"), valid_dataset.scenarios)
    save_scenario_csv(os.path.join(foldername, "test_scenarios_global.csv"), test_dataset.scenarios)
    print("[INFO] dataset/scenario summary files saved")

    # mask sanity check: 여기 값이 0이면 학습 target이 없는 상태라 바로 중단
    first_batch = next(iter(train_loader))
    removed = first_batch["observed_mask"] - first_batch["gt_mask"]
    removed_total = float(removed.sum().item())
    str_index_for_check = int(info["str_index"])
    removed_str = float(removed[:, :, str_index_for_check].sum().item())
    print(f"[CHECK] first train batch removed total points : {removed_total:.0f}")
    print(f"[CHECK] first train batch removed STr points   : {removed_str:.0f}")
    if removed_str <= 0:
        raise RuntimeError("학습용 STr global mask가 0입니다. gt_mask / scenario 설정을 확인하세요.")

    target_dim = len(info["feature_names"])
    model = CSDI_StrainGlobal(config, device, target_dim=target_dim).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] model target_dim       : {target_dim}")
    print(f"[INFO] trainable parameters   : {num_params:,}")
    print(f"[INFO] diffusion num_steps    : {config['diffusion']['num_steps']}")
    print(f"[INFO] window/stride          : {config['data']['window_size']} / {config['data']['stride']}")
    print(f"[INFO] global missing ratios  : {config['data'].get('missing_ratios')}")
    print(f"[INFO] train positions        : {config['data'].get('train_missing_positions')}")
    print(f"[INFO] test positions         : {config['data'].get('test_missing_positions')}")

    # training 시작 전 실제 CSDI loss가 0이 아닌지 확인
    with torch.no_grad():
        model.train()
        dry_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in first_batch.items()}
        dry_loss = model(dry_batch, is_train=1)
    print(f"[CHECK] dry-run train loss before epoch 1 : {float(dry_loss.item()):.8f}")
    if float(dry_loss.item()) == 0.0:
        raise RuntimeError("dry-run loss가 0입니다. 학습을 중단합니다. mask/forward를 확인하세요.")

    if args.modelfolder:
        best_path = os.path.join(foldername, "model_best.pth")
        final_path = os.path.join(foldername, "model.pth")
        load_path = best_path if args.use_best and os.path.isfile(best_path) else final_path
        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"평가용 모델 파일을 찾지 못했습니다: {load_path}")
        print(f"[LOAD] model: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location=device))
    else:
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            valid_epoch_interval=int(config["train"].get("valid_epoch_interval", 20)),
            foldername=foldername,
        )

        if args.use_best:
            best_path = os.path.join(foldername, "model_best.pth")
            if os.path.isfile(best_path):
                print(f"[LOAD] best model for final evaluation: {best_path}")
                model.load_state_dict(torch.load(best_path, map_location=device))
            else:
                print("[WARN] model_best.pth가 없어 final model로 평가합니다.")

    start = time.time()
    evaluate_and_plot(
        model,
        test_loader,
        test_dataset,
        nsample=args.nsample,
        foldername=foldername,
    )
    print(f"[DONE] strain CSDI GLOBAL block reconstruction finished | eval time={(time.time()-start)/60.0:.2f} min")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    main()
