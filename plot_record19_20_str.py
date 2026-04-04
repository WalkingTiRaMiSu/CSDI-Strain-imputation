import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

try:
    import torch
except ImportError:
    torch = None

# =========================
# 설정
# =========================
MAT_FILE = "resp_total_re_05.mat"
NORM_FILE = os.path.join("processed_05", "norm_info.npz")
SAVE_ROOT = "save"
OUTPUT_DIR = "plot_results"

WINDOW_SIZE = 256
STRIDE = 128
NSAMPLE_FILE_NAME = "generated_outputs_nsample20.pk"  # exe_nonlinear.py에서 nsample=20으로 돌렸으니까

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 공통 함수
# =========================
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


def find_latest_save_folder():
    candidates = glob.glob(os.path.join(SAVE_ROOT, "nonlinear_05_*"))
    if not candidates:
        raise FileNotFoundError("save/nonlinear_05_* 폴더를 찾지 못했습니다.")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_generated_outputs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    obj = recursive_to_cpu(obj)
    return obj


def extract_generated_array(obj):
    """
    generated_outputs_nsample20.pk는 보통 tuple/list 형태.
    첫 번째 원소가 generated samples인 경우가 많음.
    """
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("pickle 안이 비어 있습니다.")
        generated = obj[0]
        return generated

    if isinstance(obj, dict):
        for k in ["samples", "generated_samples", "all_generated_samples", "imputed_samples"]:
            if k in obj:
                return obj[k]

    raise ValueError("generated outputs를 찾지 못했습니다.")


def convert_generated_to_pred_str_windows(generated):
    """
    generated를 받아서
    최종적으로 (N_test_windows, WINDOW_SIZE) 형태의
    str prediction window 평균값을 반환.
    """
    arr = to_numpy_safe(generated)

    # object array면 안쪽 원소를 다시 쌓아보기
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            arr = np.stack([to_numpy_safe(v) for v in generated], axis=0)
        except Exception:
            raise ValueError(f"generated 배열 형태를 해석하지 못했습니다. shape={arr.shape}")

    if not isinstance(arr, np.ndarray):
        raise ValueError("generated가 numpy 배열로 변환되지 않았습니다.")

    # 가능한 경우들 처리
    # 1) (B, nsample, K, L)
    if arr.ndim == 4 and arr.shape[2] == 4:
        pred_mean = arr.mean(axis=1)      # (B, K, L)
        pred_str = pred_mean[:, 3, :]     # (B, L)
        return pred_str

    # 2) (B, nsample, L, K)
    if arr.ndim == 4 and arr.shape[3] == 4:
        pred_mean = arr.mean(axis=1)      # (B, L, K)
        pred_str = pred_mean[:, :, 3]     # (B, L)
        return pred_str

    # 3) (B, K, L)
    if arr.ndim == 3 and arr.shape[1] == 4:
        pred_str = arr[:, 3, :]           # (B, L)
        return pred_str

    # 4) (B, L, K)
    if arr.ndim == 3 and arr.shape[2] == 4:
        pred_str = arr[:, :, 3]           # (B, L)
        return pred_str

    raise ValueError(f"지원하지 않는 generated shape: {arr.shape}")


def count_windows(T, window_size=256, stride=128):
    if T < window_size:
        return 0
    return ((T - window_size) // stride) + 1


def reconstruct_from_windows(pred_windows, total_length, window_size=256, stride=128):
    """
    overlap 평균으로 full-length signal 복원
    pred_windows: (n_windows, window_size)
    return: (total_length,) with NaN on uncovered tail
    """
    full_sum = np.zeros(total_length, dtype=np.float64)
    full_cnt = np.zeros(total_length, dtype=np.float64)

    n_windows = len(pred_windows)
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        if end > total_length:
            break
        full_sum[start:end] += pred_windows[i]
        full_cnt[start:end] += 1.0

    out = np.full(total_length, np.nan, dtype=np.float64)
    valid = full_cnt > 0
    out[valid] = full_sum[valid] / full_cnt[valid]
    return out


# =========================
# 1) 원본 MAT에서 19,20번 str 불러오기
# =========================
data = loadmat(MAT_FILE)
records = data["resp_total_re_05"]

rec19 = records[0, 18]
rec20 = records[0, 19]

gt19 = np.asarray(rec19["str"]).squeeze().astype(np.float64)
gt20 = np.asarray(rec20["str"]).squeeze().astype(np.float64)

T19 = len(gt19)
T20 = len(gt20)

print("record 19 length:", T19)
print("record 20 length:", T20)

# =========================
# 2) normalization 정보
# =========================
norm = np.load(NORM_FILE)
y_mean = float(norm["y_mean"])
y_std = float(norm["y_std"])

print("y_mean:", y_mean)
print("y_std :", y_std)

# =========================
# 3) latest save 폴더에서 generated outputs 읽기
# =========================
latest_folder = find_latest_save_folder()
gen_file = os.path.join(latest_folder, NSAMPLE_FILE_NAME)

if not os.path.isfile(gen_file):
    raise FileNotFoundError(f"{gen_file} 파일을 찾지 못했습니다.")

print("Using generated file:", gen_file)

obj = load_generated_outputs(gen_file)
generated = extract_generated_array(obj)
pred_str_windows_norm = convert_generated_to_pred_str_windows(generated)

print("pred_str_windows_norm shape:", pred_str_windows_norm.shape)

# =========================
# 4) test windows를 19/20으로 분리
#    make_windows.py에서 test는 19 -> 20 순서로 저장됨
# =========================
n19 = count_windows(T19, WINDOW_SIZE, STRIDE)
n20 = count_windows(T20, WINDOW_SIZE, STRIDE)

print("record 19 window count:", n19)
print("record 20 window count:", n20)

if pred_str_windows_norm.shape[0] < n19 + n20:
    raise ValueError(
        f"예측 window 수가 부족합니다. pred={pred_str_windows_norm.shape[0]}, needed={n19+n20}"
    )

pred19_norm = pred_str_windows_norm[:n19]
pred20_norm = pred_str_windows_norm[n19:n19+n20]

# =========================
# 5) full curve로 복원
# =========================
pred19_norm_full = reconstruct_from_windows(pred19_norm, T19, WINDOW_SIZE, STRIDE)
pred20_norm_full = reconstruct_from_windows(pred20_norm, T20, WINDOW_SIZE, STRIDE)

# denormalize
pred19 = pred19_norm_full * y_std + y_mean
pred20 = pred20_norm_full * y_std + y_mean

# =========================
# 6) plot 저장
# =========================
def save_plot(gt, pred, record_num, save_name):
    x = np.arange(len(gt))

    plt.figure(figsize=(14, 5))
    plt.plot(x, gt, label="Ground Truth")
    plt.plot(x, pred, label="Prediction")
    plt.xlabel("Time Index")
    plt.ylabel("Strain")
    plt.title(f"Record {record_num}: Ground Truth vs Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_name, dpi=200)
    plt.show()
    print(f"Saved: {save_name}")

save_plot(gt19, pred19, 19, os.path.join(OUTPUT_DIR, "record19_gt_vs_pred.png"))
save_plot(gt20, pred20, 20, os.path.join(OUTPUT_DIR, "record20_gt_vs_pred.png"))

print("\nDone.")