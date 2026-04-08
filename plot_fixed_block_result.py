import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

try:
    import torch
except ImportError:
    torch = None

from fixed_mask_settings import (
    EVAL_SAVE_FOLDER_NAME,
    NSAMPLE,
    FS,
    WINDOW_SIZE,
    STRIDE,
    MASK_19_START, MASK_19_END,
    MASK_20_START, MASK_20_END,
)

MAT_FILE = "resp_total_re_05.mat"
NORM_FILE = os.path.join("processed_str_05", "norm_info.npz")
SAVE_ROOT = "save"
OUTPUT_DIR = "plot_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def recursive_to_cpu(obj):
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: recursive_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [recursive_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(recursive_to_cpu(v) for v in obj)
    return obj


def to_numpy_safe(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, list):
        return np.array([to_numpy_safe(v) for v in x], dtype=object)
    if isinstance(x, tuple):
        return np.array([to_numpy_safe(v) for v in x], dtype=object)
    return x


def load_generated_outputs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return recursive_to_cpu(obj)


def extract_generated_array(obj):
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("pickle 안이 비어 있습니다.")
        return obj[0]

    if isinstance(obj, dict):
        for k in ["samples", "generated_samples", "all_generated_samples", "imputed_samples"]:
            if k in obj:
                return obj[k]

    raise ValueError("generated outputs를 찾지 못했습니다.")


def convert_generated_to_pred_windows(generated):
    arr = to_numpy_safe(generated)

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = np.stack([to_numpy_safe(v) for v in generated], axis=0)
        except Exception:
            raise ValueError(f"generated 배열 형태를 해석하지 못했습니다. shape={arr.shape}")

    if not isinstance(arr, np.ndarray):
        raise ValueError("generated가 numpy 배열로 변환되지 않았습니다.")

    # (B, nsample, 1, L)
    if arr.ndim == 4 and arr.shape[2] == 1:
        pred_mean = arr.mean(axis=1)
        return pred_mean[:, 0, :]

    # (B, nsample, L, 1)
    if arr.ndim == 4 and arr.shape[3] == 1:
        pred_mean = arr.mean(axis=1)
        return pred_mean[:, :, 0]

    # (B, 1, L)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]

    # (B, L, 1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]

    raise ValueError(f"지원하지 않는 generated shape: {arr.shape}")


def build_mask(length, start, end):
    mask = np.ones(length, dtype=np.float32)
    mask[start:end] = 0.0
    return mask


def get_overlap_window_starts(length, missing_start, missing_end):
    starts = []
    start = 0
    while start + WINDOW_SIZE <= length:
        end = start + WINDOW_SIZE
        overlaps = not (end <= missing_start or start >= missing_end)
        if overlaps:
            starts.append(start)
        start += STRIDE
    return starts


def reconstruct_missing_only(pred_windows_norm, starts, total_length, mask_full):
    """
    결측 위치만 prediction 평균으로 복원
    관측 위치는 NaN으로 남겨두고 나중에 GT를 그대로 넣음
    """
    full_sum = np.zeros(total_length, dtype=np.float64)
    full_cnt = np.zeros(total_length, dtype=np.float64)

    for pred_win, start in zip(pred_windows_norm, starts):
        end = start + WINDOW_SIZE
        local_missing = (mask_full[start:end] == 0)

        full_sum[start:end][local_missing] += pred_win[local_missing]
        full_cnt[start:end][local_missing] += 1.0

    out = np.full(total_length, np.nan, dtype=np.float64)
    valid = full_cnt > 0
    out[valid] = full_sum[valid] / full_cnt[valid]
    return out


def save_plot(gt, final_recon, rec_num, miss_start, miss_end, save_path):
    time_s = np.arange(len(gt)) / FS

    plt.figure(figsize=(14, 5))
    plt.axvspan(miss_start / FS, miss_end / FS, color="gray", alpha=0.15, zorder=0)
    plt.plot(time_s, gt, label="Ground Truth", linewidth=2.0)
    plt.plot(time_s, final_recon, label="Reconstruction", linewidth=2.0)

    plt.xlabel("Time (s)")
    plt.ylabel("Strain")
    plt.legend(frameon=False)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    # 1) 원본 GT
    data = loadmat(MAT_FILE)
    records = data["resp_total_re_05"]

    gt19 = np.asarray(records[0, 18]["str"]).squeeze().astype(np.float64)
    gt20 = np.asarray(records[0, 19]["str"]).squeeze().astype(np.float64)

    T19 = len(gt19)
    T20 = len(gt20)

    # 2) 정규화 정보
    norm = np.load(NORM_FILE)
    y_mean = float(norm["y_mean"])
    y_std = float(norm["y_std"])

    # 3) 평가 결과 불러오기
    eval_folder = os.path.join(SAVE_ROOT, EVAL_SAVE_FOLDER_NAME)
    gen_file = os.path.join(eval_folder, f"generated_outputs_nsample{NSAMPLE}.pk")

    if not os.path.exists(gen_file):
        raise FileNotFoundError(f"파일 없음: {gen_file}")

    obj = load_generated_outputs(gen_file)
    generated = extract_generated_array(obj)
    pred_windows_norm = convert_generated_to_pred_windows(generated)

    # 4) 19 / 20 윈도우 시작점 재구성
    starts19 = get_overlap_window_starts(T19, MASK_19_START, MASK_19_END)
    starts20 = get_overlap_window_starts(T20, MASK_20_START, MASK_20_END)

    n19 = len(starts19)
    n20 = len(starts20)

    if pred_windows_norm.shape[0] < n19 + n20:
        raise ValueError(
            f"예측 window 수 부족: pred={pred_windows_norm.shape[0]}, needed={n19+n20}"
        )

    pred19_norm = pred_windows_norm[:n19]
    pred20_norm = pred_windows_norm[n19:n19+n20]

    # 5) mask 생성
    mask19 = build_mask(T19, MASK_19_START, MASK_19_END)
    mask20 = build_mask(T20, MASK_20_START, MASK_20_END)

    # 6) 결측 부분만 복원
    pred19_missing_norm = reconstruct_missing_only(pred19_norm, starts19, T19, mask19)
    pred20_missing_norm = reconstruct_missing_only(pred20_norm, starts20, T20, mask20)

    # 7) GT normalized 만들기
    gt19_norm = (gt19 - y_mean) / y_std
    gt20_norm = (gt20 - y_mean) / y_std

    # 8) 최종 복원: 관측 구간은 GT, 결측 구간만 prediction
    final19_norm = gt19_norm.copy()
    final20_norm = gt20_norm.copy()

    missing19 = (mask19 == 0)
    missing20 = (mask20 == 0)

    final19_norm[missing19] = pred19_missing_norm[missing19]
    final20_norm[missing20] = pred20_missing_norm[missing20]

    # 9) denormalize
    final19 = final19_norm * y_std + y_mean
    final20 = final20_norm * y_std + y_mean

    # 10) plot 저장
    save_plot(
        gt19,
        final19,
        19,
        MASK_19_START,
        MASK_19_END,
        os.path.join(OUTPUT_DIR, f"record19_{EVAL_SAVE_FOLDER_NAME}_fixed_block.png")
    )

    save_plot(
        gt20,
        final20,
        20,
        MASK_20_START,
        MASK_20_END,
        os.path.join(OUTPUT_DIR, f"record20_{EVAL_SAVE_FOLDER_NAME}_fixed_block.png")
    )

    print("\nDone.")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18
    main()