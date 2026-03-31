import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import glob

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["axes.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["lines.linewidth"] = 1.5


def get_feature_columns(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if str(c).startswith("ch_")]
    if len(feature_cols) == 0:
        raise ValueError(f"ch_ 로 시작하는 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
    return feature_cols


folders = sorted(glob.glob("./save/custom_*"))
if len(folders) == 0:
    raise ValueError("save/custom_* 폴더가 없습니다.")

folder = folders[-1]
print("사용하는 결과 폴더:", folder)

pk_path = os.path.join(folder, "generated_outputs_nsample20.pk")
original_csv = "./custom_data/original.csv"
missing_csv = "./custom_data/missing.csv"

with open(pk_path, "rb") as f:
    data = pickle.load(f)

samples = data[0]
if torch.is_tensor(samples):
    samples = samples.cpu().numpy()

# samples: (1, nsample, T, K)
pred_samples = samples[0]   # (nsample, T, K)

df_org = pd.read_csv(original_csv)
df_mis = pd.read_csv(missing_csv)

feature_cols = get_feature_columns(df_org)
if "ch_8" not in feature_cols:
    raise ValueError(f"ch_8이 없습니다. feature_cols={feature_cols}")

ch8_idx = feature_cols.index("ch_8")

org = df_org[feature_cols].astype(float).values
mis = df_mis[feature_cols].astype(float).values

# original 기준 정규화
mean = np.zeros(org.shape[1], dtype=np.float32)
std = np.zeros(org.shape[1], dtype=np.float32)

for k in range(org.shape[1]):
    valid = ~np.isnan(org[:, k])
    mean[k] = org[valid, k].mean()
    std[k] = org[valid, k].std()
    if std[k] == 0 or np.isnan(std[k]):
        std[k] = 1.0

true_norm = np.zeros_like(org, dtype=np.float32)
for k in range(org.shape[1]):
    true_norm[:, k] = (org[:, k] - mean[k]) / std[k]

# ch8만 선택
true_ch8 = true_norm[:, ch8_idx]
pred_ch8_samples = pred_samples[:, :, ch8_idx]   # (nsample, T)

pred_median = np.median(pred_ch8_samples, axis=0)
pred_low = np.percentile(pred_ch8_samples, 5, axis=0)
pred_high = np.percentile(pred_ch8_samples, 95, axis=0)

given_mask = ~np.isnan(mis[:, ch8_idx])
missing_mask = np.isnan(mis[:, ch8_idx])

t = np.arange(len(true_ch8))

print("feature_cols:", feature_cols)
print("ch8 index:", ch8_idx)
print("데이터 길이:", len(true_ch8))
print("결측 개수:", int(missing_mask.sum()))

missing_idx = np.where(missing_mask)[0]
segments = []

if len(missing_idx) > 0:
    start = missing_idx[0]
    prev = missing_idx[0]
    for idx in missing_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            segments.append((start, prev))
            start = idx
            prev = idx
    segments.append((start, prev))

print("\n결측 구간:")
for i, (s, e) in enumerate(segments, 1):
    print(f"{i:02d}: {s} ~ {e} (길이 {e-s+1})")

# 전체 그림
median_plot = np.full_like(pred_median, np.nan, dtype=float)
low_plot = np.full_like(pred_low, np.nan, dtype=float)
high_plot = np.full_like(pred_high, np.nan, dtype=float)

median_plot[missing_mask] = pred_median[missing_mask]
low_plot[missing_mask] = pred_low[missing_mask]
high_plot[missing_mask] = pred_high[missing_mask]

plt.figure(figsize=(16, 5))

plt.plot(t, true_ch8, color="royalblue", linewidth=1.0, label="Ground Truth")
plt.scatter(
    t[given_mask], true_ch8[given_mask],
    color="tomato", marker="x", s=18, linewidths=0.8,
    label="Given (observed)"
)
plt.scatter(
    t[missing_mask], true_ch8[missing_mask],
    color="royalblue", s=10,
    label="Target (missing)"
)
plt.plot(
    t, median_plot,
    color="forestgreen", linewidth=1.8,
    label="CSDI (median)"
)
plt.fill_between(
    t, low_plot, high_plot,
    color="forestgreen", alpha=0.20,
    label="90% CI"
)

plt.title("CSDI Imputation Result for ch_8 (Full)")
plt.xlabel("Time Step")
plt.ylabel("Normalized Value")
plt.grid(True, alpha=0.22)
plt.legend(loc="best", frameon=True)
plt.tight_layout()

full_path = os.path.join(folder, "multichannel_ch8_full.png")
plt.savefig(full_path, dpi=220, bbox_inches="tight")
plt.show()

print("\n저장:", full_path)

# 확대 그림
for i, (s, e) in enumerate(segments, 1):
    pad = 70
    left = max(0, s - pad)
    right = min(len(true_ch8) - 1, e + pad)

    x = t[left:right+1]
    y_true = true_ch8[left:right+1]
    y_med = pred_median[left:right+1]
    y_low = pred_low[left:right+1]
    y_high = pred_high[left:right+1]

    local_given = given_mask[left:right+1]
    local_missing = missing_mask[left:right+1]

    y_med_plot = np.full_like(y_med, np.nan, dtype=float)
    y_low_plot = np.full_like(y_low, np.nan, dtype=float)
    y_high_plot = np.full_like(y_high, np.nan, dtype=float)

    y_med_plot[local_missing] = y_med[local_missing]
    y_low_plot[local_missing] = y_low[local_missing]
    y_high_plot[local_missing] = y_high[local_missing]

    plt.figure(figsize=(11, 4.2))

    plt.plot(x, y_true, color="royalblue", linewidth=1.3, label="Ground Truth")
    plt.scatter(
        x[local_given], y_true[local_given],
        color="tomato", marker="x", s=22, linewidths=0.8,
        label="Given (observed)"
    )
    plt.scatter(
        x[local_missing], y_true[local_missing],
        color="royalblue", s=14,
        label="Target (missing)"
    )
    plt.plot(
        x, y_med_plot,
        color="forestgreen", linewidth=2.0,
        label="CSDI (median)"
    )
    plt.fill_between(
        x, y_low_plot, y_high_plot,
        color="forestgreen", alpha=0.20,
        label="90% CI"
    )
    plt.axvspan(s, e, color="khaki", alpha=0.20)

    plt.title(f"Multichannel Zoomed Imputation Result for ch_8 ({s}~{e})")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Value")
    plt.grid(True, alpha=0.22)
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()

    save_path = os.path.join(folder, f"multichannel_ch8_zoom_{i:02d}.png")
    plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()

    print("저장:", save_path)