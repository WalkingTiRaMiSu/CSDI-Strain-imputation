import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
df_org = pd.read_csv("custom_data/original.csv")
df_mis = pd.read_csv("custom_data/missing.csv")

org = df_org["ch_1"].values
mis = df_mis["ch_1"].values

# 결손 위치
mask = np.isnan(mis)

# 간단한 보간 (비교용 baseline)
interp = pd.Series(mis).interpolate().fillna(method="bfill").fillna(method="ffill").values

# 그래프
plt.figure(figsize=(12,6))

plt.plot(org, label="Ground Truth", linewidth=2)
plt.plot(mis, 'o', label="Observed (Missing)", alpha=0.6)
plt.plot(interp, '--', label="Interpolation (baseline)")

plt.legend()
plt.title("Time Series Imputation (CH1)")
plt.xlabel("Time")
plt.ylabel("Value")

plt.show()