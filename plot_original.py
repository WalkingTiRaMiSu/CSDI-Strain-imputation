import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("custom_data/original.csv")

data = df["ch_1"].values

plt.figure(figsize=(12,6))
plt.plot(data, linewidth=2)

plt.title("Original Time Series (CH1)")
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
plt.show()