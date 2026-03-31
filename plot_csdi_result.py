import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# 🔥 여기에 네 폴더 이름 넣기
folder = "save/custom_20260324_195358/"

df_org = pd.read_csv("custom_data/original.csv")
df_mis = pd.read_csv("custom_data/missing.csv")

value_cols = df_org.columns[1:]

org = df_org[value_cols].astype(float).values
mis = df_mis[value_cols].astype(float).values
mask = np.isnan(mis)

# mean / std 계산
mean = np.zeros(org.shape[1])
std = np.zeros(org.shape[1])

for k in range(org.shape[1]):
    valid = ~np.isnan(org[:, k])
    mean[k] = org[valid, k].mean()
    std[k] = org[valid, k].std()
    if std[k] == 0 or np.isnan(std[k]):
        std[k] = 1.0

# 🔥 자동으로 nsample 파일 찾기
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

# 평균 예측
if samples.ndim == 4:
    if samples.shape[0] == 1:
        pred = samples[0].mean(axis=0)
    elif samples.shape[1] == 1:
        pred = samples[:, 0].mean(axis=0)
    else:
        pred = samples.mean(axis=0)[0]
elif samples.ndim == 3:
    pred = samples.mean(axis=0)
else:
    pred = samples

# 길이 맞추기
min_len = min(org.shape[0], mis.shape[0], pred.shape[0])

org = org[:min_len]
mis = mis[:min_len]
pred = pred[:min_len]
mask = mask[:min_len]

# 역정규화
for k in range(pred.shape[1]):
    pred[:, k] = pred[:, k] * std[k] + mean[k]

# 그래프
fig, axes = plt.subplots(len(value_cols), 1, figsize=(12, 3*len(value_cols)), sharex=True)

if len(value_cols) == 1:
    axes = [axes]

for i, col in enumerate(value_cols):
    observed_only = mis[:, i].copy()

    pred_missing_only = np.full_like(pred[:, i], np.nan)
    pred_missing_only[mask[:, i]] = pred[:, i][mask[:, i]]

    axes[i].plot(org[:, i], label="Ground Truth", color="blue")
    axes[i].plot(observed_only, "o", label="Observed", alpha=0.6, color="orange")
    axes[i].plot(pred_missing_only, "--", label="CSDI Imputation", color="green")

    axes[i].fill_between(
        np.arange(min_len),
        np.min(org[:, i]),
        np.max(org[:, i]),
        where=mask[:, i],
        color="gray",
        alpha=0.2
    )

    axes[i].set_title(f"{col} Imputation Result")
    axes[i].legend()

plt.xlabel("Time")
plt.tight_layout()
plt.show()