import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 가장 최근 save 폴더 찾기
# -----------------------------
save_folders = sorted(glob.glob("./save/custom_*"))
if len(save_folders) == 0:
    raise ValueError("save/custom_* 폴더가 없습니다. 먼저 exe_custom.py를 실행하세요.")

latest_folder = save_folders[-1]
print("가장 최근 결과 폴더:", latest_folder)

# -----------------------------
# 저장된 결과 불러오기
# -----------------------------
samples = np.load(os.path.join(latest_folder, "samples.npy"))          # (B, nsample, K, L)
observed_data = np.load(os.path.join(latest_folder, "observed_data.npy"))  # (B, K, L)
target_mask = np.load(os.path.join(latest_folder, "target_mask.npy"))      # (B, K, L)
observed_mask = np.load(os.path.join(latest_folder, "observed_mask.npy"))  # (B, K, L)
observed_tp = np.load(os.path.join(latest_folder, "observed_tp.npy"))      # (B, L)

# 원본 csv도 불러오기
df_org = pd.read_csv("./custom_data/original.csv")
value_cols = df_org.columns[1:]
org = df_org[value_cols].astype(float).values

# -----------------------------
# 첫 번째 샘플, 첫 번째 채널(ch8만 있으면 채널 0)
# -----------------------------
b = 0
k = 0

sample_paths = samples[b, :, k, :]       # (nsample, L)
obs_data = observed_data[b, k, :]        # (L,)
tmask = target_mask[b, k, :]             # (L,)
omask = observed_mask[b, k, :]           # (L,)
tp = observed_tp[b]                      # (L,)

# original.csv에서 실제 정답 불러오기
true_values = org[:, k]

# 예측 통계
median = np.median(sample_paths, axis=0)
lower = np.percentile(sample_paths, 5, axis=0)
upper = np.percentile(sample_paths, 95, axis=0)

# observed / missing 위치
given_mask = (omask - tmask) > 0     # 관측된 위치
missing_mask = tmask > 0             # 복원 대상 위치

# -----------------------------
# 전체 구간 그래프
# -----------------------------
plt.figure(figsize=(16, 5))
plt.plot(tp, true_values, label="Ground Truth", linewidth=1.5)
plt.scatter(tp[given_mask], true_values[given_mask], s=12, label="Observed")
plt.plot(tp, median, label="CSDI Median", linewidth=2)
plt.fill_between(tp, lower, upper, alpha=0.25, label="90% CI")
plt.scatter(tp[missing_mask], true_values[missing_mask], s=14, label="Missing Target")

plt.title("CSDI Imputation Result - Full Range")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()

full_png = os.path.join(latest_folder, "result_full.png")
plt.savefig(full_png, dpi=200)
plt.show()

print("전체 그래프 저장:", full_png)

# -----------------------------
# 결측 구간들 찾기
# -----------------------------
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

print("결측 구간 목록:")
for i, (s, e) in enumerate(segments, 1):
    print(f"{i:02d}: {s} ~ {e}")

# -----------------------------
# 각 결측 구간 확대 그래프 저장
# -----------------------------
for i, (s, e) in enumerate(segments, 1):
    pad = 80
    left = max(0, s - pad)
    right = min(len(tp) - 1, e + pad)

    x = tp[left:right+1]
    y_true = true_values[left:right+1]
    y_med = median[left:right+1]
    y_low = lower[left:right+1]
    y_up = upper[left:right+1]

    local_given = given_mask[left:right+1]
    local_missing = missing_mask[left:right+1]

    plt.figure(figsize=(14, 5))
    plt.plot(x, y_true, label="Ground Truth", linewidth=1.5)
    plt.scatter(x[local_given], y_true[local_given], s=18, label="Observed")
    plt.plot(x, y_med, label="CSDI Median", linewidth=2)
    plt.fill_between(x, y_low, y_up, alpha=0.25, label="90% CI")
    plt.scatter(x[local_missing], y_true[local_missing], s=22, label="Missing Target")

    plt.axvspan(tp[s], tp[e], alpha=0.15, label="Missing Block")
    plt.title(f"CSDI Imputation Result - Segment {i} ({s}~{e})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    seg_png = os.path.join(latest_folder, f"result_segment_{i:02d}.png")
    plt.savefig(seg_png, dpi=200)
    plt.show()

    print("구간 그래프 저장:", seg_png)