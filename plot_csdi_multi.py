import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# =========================
# 1. 여기만 수정
# =========================
folder = "save/custom_20260326_025058/"   # 예: save/custom_20260327_012345/
missing_start = 4000
missing_end = 7000
view_start = 3500
view_end = 7500

# =========================
# 2. 데이터 불러오기
# =========================
df_org = pd.read_csv("custom_data/original.csv")
df_mis = pd.read_csv("custom_data/missing.csv")

value_cols = df_org.columns[1:]   # ch_1 ~ ch_16

# sentinel 값 처리
df_org[value_cols] = df_org[value_cols].replace(-1000000, np.nan)
df_mis[value_cols] = df_mis[value_cols].replace(-1000000, np.nan)

org = df_org[value_cols].astype(float).values
mis = df_mis[value_cols].astype(float).values
mask = np.isnan(mis)

# =========================
# 3. CSDI 결과 불러오기
# =========================
files = os.listdir(folder)
pk_files = [f for f in files if f.startswith("generated_outputs") and f.endswith(".pk")]
if len(pk_files) == 0:
    raise FileNotFoundError("generated_outputs 파일 없음")

print("사용 파일:", pk_files[0])

with open(os.path.join(folder, pk_files[0]), "rb") as f:
    data = pickle.load(f)

samples = data[0]
if hasattr(samples, "cpu"):
    samples = samples.cpu().numpy()
samples = np.array(samples)

if samples.ndim == 4:
    if samples.shape[0] == 1:        # (B, nsample, T, K)
        pred = samples[0].mean(axis=0)
    elif samples.shape[1] == 1:      # (nsample, B, T, K)
        pred = samples[:, 0].mean(axis=0)
    else:
        pred = samples.mean(axis=0)[0]
elif samples.ndim == 3:
    pred = samples.mean(axis=0)
elif samples.ndim == 2:
    pred = samples
else:
    raise ValueError(f"Unexpected samples shape: {samples.shape}")

pred = np.array(pred)

# =========================
# 4. 길이 맞추기
# =========================
min_len = min(org.shape[0], mis.shape[0], pred.shape[0])
org = org[:min_len]
mis = mis[:min_len]
pred = pred[:min_len]
mask = mask[:min_len]

# =========================
# 5. pred 역정규화
# =========================
mean = np.zeros(org.shape[1], dtype=np.float32)
std = np.zeros(org.shape[1], dtype=np.float32)

for k in range(org.shape[1]):
    mean[k] = np.nanmean(org[:, k])
    std[k] = np.nanstd(org[:, k])
    if std[k] == 0 or np.isnan(std[k]):
        std[k] = 1.0

for k in range(pred.shape[1]):
    pred[:, k] = pred[:, k] * std[k] + mean[k]

# =========================
# 6. 구간 잘라서 그리기
# =========================
view_start = max(0, view_start)
view_end = min(min_len, view_end)
x = np.arange(view_start, view_end)

def plot_group(start_ch, end_ch, fig_title):
    n = end_ch - start_ch
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.0 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(fig_title, fontsize=14)

    for plot_idx, ch_idx in enumerate(range(start_ch, end_ch)):
        ax = axes[plot_idx]

        org_seg = org[view_start:view_end, ch_idx]
        mis_seg = mis[view_start:view_end, ch_idx]
        pred_seg = pred[view_start:view_end, ch_idx]
        mask_seg = mask[view_start:view_end, ch_idx]

        # 원래값
        ax.plot(x, org_seg, color="blue", linewidth=1.2, label="Ground Truth")

        # 관측값
        ax.plot(x, mis_seg, color="orange", linewidth=0.8, alpha=0.8, label="Observed")

        # 결측구간에서만 복원값
        pred_missing_only = np.full_like(pred_seg, np.nan)
        pred_missing_only[mask_seg] = pred_seg[mask_seg]
        ax.plot(
            x,
            pred_missing_only,
            color="green",
            linestyle="--",
            linewidth=1.1,
            label="CSDI Imputation"
        )

        # 결측구간 회색 표시
        ax.axvspan(missing_start, missing_end, color="gray", alpha=0.18)

        # y축: original 기준
        valid_org = org_seg[~np.isnan(org_seg)]
        if len(valid_org) > 0:
            y_min = np.min(valid_org)
            y_max = np.max(valid_org)
            margin = 0.20 * (y_max - y_min + 1e-8)
            if margin == 0:
                margin = 1.0
            ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_title(value_cols[ch_idx], fontsize=10)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Time Index")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

# 총 8장: 2채널씩
plot_group(0, 2, "Channels 1-2")
plot_group(2, 4, "Channels 3-4")
plot_group(4, 6, "Channels 5-6")
plot_group(6, 8, "Channels 7-8")
plot_group(8, 10, "Channels 9-10")
plot_group(10, 12, "Channels 11-12")
plot_group(12, 14, "Channels 13-14")
plot_group(14, 16, "Channels 15-16")

plt.show()