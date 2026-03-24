import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# 실행 후 생성된 최신 폴더명으로 바꿔야 함
folder = "save/custom_YYYYMMDD_HHMMSS/"

df_org = pd.read_csv("custom_data/original.csv")
df_mis = pd.read_csv("custom_data/missing.csv")

value_cols = df_org.columns[1:]

org = df_org[value_cols].astype(float).values
mis = df_mis[value_cols].astype(float).values
mask = np.isnan(mis)

# 채널별 mean/std
mean = np.zeros(org.shape[1], dtype=np.float32)
std = np.zeros(org.shape[1], dtype=np.float32)
for k in range(org.shape[1]):
    valid = ~np.isnan(org[:, k])
    mean[k] = org[valid, k].mean()
    std[k] = org[valid, k].std()
    if std[k] == 0 or np.isnan(std[k]):
        std[k] = 1.0

with open(os.path.join(folder, "generated_outputs_nsample100.pk"), "rb") as f:
    data = pickle.load(f)

samples = data[0]
if hasattr(samples, "cpu"):
    samples = samples.cpu().numpy()
samples = np.array(samples)

# 평균 예측
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

# 길이 맞춤
min_len = min(org.shape[0], mis.shape[0], pred.shape[0])
org = org[:min_len]
mis = mis[:min_len]
pred = pred[:min_len]
mask = mask[:min_len]

# 역정규화
for k in range(pred.shape[1]):
    pred[:, k] = pred[:, k] * std[k] + mean[k]

fig, axes = plt.subplots(len(value_cols), 1, figsize=(12, 3*len(value_cols)), sharex=True)

if len(value_cols) == 1:
    axes = [axes]

for i, col in enumerate(value_cols):
    observed_only = mis[:, i].copy()
    pred_missing_only = np.full_like(pred[:, i], np.nan)
    pred_missing_only[mask[:, i]] = pred[:, i][mask[:, i]]

    axes[i].plot(org[:, i], label="Ground Truth", linewidth=2, color="blue")
    axes[i].plot(observed_only, "o", label="Observed", alpha=0.6, color="orange")
    axes[i].plot(pred_missing_only, "--", label="CSDI Imputation", linewidth=2, color="green")

    axes[i].fill_between(
        np.arange(min_len),
        np.min(org[:, i]),
        np.max(org[:, i]),
        where=mask[:, i],
        color="gray",
        alpha=0.2,
        label="Missing Region"
    )

    axes[i].set_title(f"{col} Imputation Result")
    axes[i].set_ylabel("Value")
    axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()