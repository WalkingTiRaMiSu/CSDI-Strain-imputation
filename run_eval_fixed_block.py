import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import torch

from main_model import CSDI_Physio
from utils import evaluate
from dataset_test_fixed_block import get_test_dataloader
from fixed_mask_settings import (
    TRAINED_SAVE_FOLDER_NAME,
    EVAL_SAVE_FOLDER_NAME,
    NSAMPLE,
)

def main():
    trained_folder = os.path.join("save", TRAINED_SAVE_FOLDER_NAME)
    eval_folder = os.path.join("save", EVAL_SAVE_FOLDER_NAME)

    if not os.path.exists(trained_folder):
        raise FileNotFoundError(f"학습된 폴더를 찾지 못했습니다: {trained_folder}")

    os.makedirs(eval_folder, exist_ok=True)

    config_path = os.path.join(trained_folder, "config.json")
    model_path = os.path.join(trained_folder, "model.pth")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json이 없습니다: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pth가 없습니다: {model_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    device = "cuda:0"

    test_loader, dataset = get_test_dataloader(
        data_dir="./processed_str_05",
        batch_size=config["train"]["batch_size"],
    )

    model = CSDI_Physio(config, device, target_dim=1).to(device)
    model.load_state_dict(torch.load(model_path))

    # 평가만 수행
    evaluate(
        model,
        test_loader,
        nsample=NSAMPLE,
        scaler=1,
        foldername=eval_folder + "/",
    )

    # 설정 저장
    with open(os.path.join(eval_folder, "fixed_eval_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"trained_folder={TRAINED_SAVE_FOLDER_NAME}\n")
        f.write(f"nsample={NSAMPLE}\n")
        f.write(f"record19_windows={dataset.n19}\n")
        f.write(f"record20_windows={dataset.n20}\n")

    print(f"Saved evaluation results in: {eval_folder}")

if __name__ == "__main__":
    main()