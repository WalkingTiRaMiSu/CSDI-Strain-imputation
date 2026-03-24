import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("custom_data/missing.csv")

data = df["ch_1"].values
mask = np.isnan(data)

plt.figure(figsize=(12,6))

plt.plot(data, label="Observed Data", linewidth=2)
plt.scatter(np.where(mask)[0], [data.mean()]*sum(mask),
            color='red', label="Missing Points", s=20)

plt.fill_between(
    range(len(data)),
    min(data),
    max(data),
    where=mask,
    color="gray",
    alpha=0.2,
    label="Missing Region"
)

plt.legend()
plt.title("Missing Time Series (CH1)")
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
plt.show()