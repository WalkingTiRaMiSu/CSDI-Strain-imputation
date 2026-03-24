import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

folder = "save/custom_20260324_121532/"

# 원본/결손 데이터 불러오기
df_org = pd.read_csv("custom_data/original.csv")
df_mis = pd.read_csv("custom_data/missing.csv")

org = df_org["ch_1"].values.astype(float)
mis = df_mis["ch_1"].values.astype(float)

# 결손 위치
mask = np.isnan(mis)

# 정규화에 사용한 mean, std
valid = ~np.isnan(org)
mean = org[valid].mean()
std = org[valid].std()
if std == 0:
    std = 1.0

# 결과 파일 읽기
with open(os.path.join(folder, "generated_outputs_nsample100.pk"), "rb") as f:
    data = pickle.load(f)

samples = data[0]

if hasattr(samples, "cpu"):
    samples = samples.cpu().numpy()

samples = np.array(samples)

# 평균 예측값 계산
if samples.ndim == 4:
    if samples.shape[0] == 1:
        pred = samples[0].mean(axis=0).squeeze()
    else:
        pred = samples.mean(axis=0).squeeze()
elif samples.ndim == 3:
    pred = samples.mean(axis=0).squeeze()
elif samples.ndim == 2:
    pred = samples.squeeze()
else:
    raise ValueError(f"Unexpected samples shape: {samples.shape}")

pred = np.array(pred).reshape(-1)

# 길이 맞추기
min_len = min(len(org), len(pred), len(mis))
org = org[:min_len]
mis = mis[:min_len]
pred = pred[:min_len]
mask = mask[:min_len]

# 역정규화
pred = pred * std + mean

# 관측값은 결손 아닌 부분만 남김
observed_only = mis.copy()

# ✅ CSDI 예측은 결손 구간에서만 보이게
pred_missing_only = np.full_like(pred, np.nan)
pred_missing_only[mask] = pred[mask]

plt.figure(figsize=(12, 6))

plt.plot(org, label="Ground Truth", linewidth=2, color="blue")
plt.plot(observed_only, "o", label="Observed", alpha=0.6, color="orange")
plt.plot(pred_missing_only, "--", label="CSDI Imputation", linewidth=2, color="green")

plt.fill_between(
    np.arange(len(org)),
    np.min(org),
    np.max(org),
    where=mask,
    color="gray",
    alpha=0.2,
    label="Missing Region"
)

plt.legend()
plt.title("CSDI Imputation Result (CH1)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.tight_layout()
plt.show()