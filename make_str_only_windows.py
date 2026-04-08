from scipy.io import loadmat
import numpy as np
import os

MAT_FILE = "resp_total_re_05.mat"
SAVE_DIR = "processed_str_05"

WINDOW_SIZE = 256
STRIDE = 128

os.makedirs(SAVE_DIR, exist_ok=True)

data = loadmat(MAT_FILE)
records = data["resp_total_re_05"]

all_str = []

print("=" * 60)
print("Load strain-only dataset")
print("=" * 60)

for i in range(20):
    rec = records[0, i]
    strr = np.asarray(rec["str"]).squeeze().astype(np.float32)
    all_str.append(strr)
    print(f"Record {i+1:02d}: length = {len(strr)}")

train_list = all_str[:18]   # 1~18
test_list = all_str[18:]    # 19~20

train_concat = np.concatenate(train_list, axis=0)
y_mean = train_concat.mean()
y_std = train_concat.std() + 1e-8

def normalize_y(y):
    return ((y - y_mean) / y_std).astype(np.float32)

train_list = [normalize_y(y) for y in train_list]
test_list = [normalize_y(y) for y in test_list]

def make_windows_1d(y, window_size=256, stride=128):
    windows = []
    T = len(y)
    start = 0
    while start + window_size <= T:
        end = start + window_size
        windows.append(y[start:end])
        start += stride
    return np.array(windows, dtype=np.float32)

train_windows = []
for i, y in enumerate(train_list, start=1):
    w = make_windows_1d(y, WINDOW_SIZE, STRIDE)
    train_windows.append(w)
    print(f"Train record {i:02d} -> {len(w)} windows")

test_windows = []
for i, y in enumerate(test_list, start=19):
    w = make_windows_1d(y, WINDOW_SIZE, STRIDE)
    test_windows.append(w)
    print(f"Test  record {i:02d} -> {len(w)} windows")

train_windows = np.concatenate(train_windows, axis=0)
test_windows = np.concatenate(test_windows, axis=0)

print("=" * 60)
print("Window summary")
print("train_windows:", train_windows.shape)
print("test_windows :", test_windows.shape)
print("=" * 60)

np.save(os.path.join(SAVE_DIR, "train_windows.npy"), train_windows)
np.save(os.path.join(SAVE_DIR, "test_windows.npy"), test_windows)
np.savez(
    os.path.join(SAVE_DIR, "norm_info.npz"),
    y_mean=y_mean,
    y_std=y_std,
    window_size=WINDOW_SIZE,
    stride=STRIDE,
)

# 원본 normalized full record도 저장 (나중에 full reconstruction에 필요)
np.save(os.path.join(SAVE_DIR, "record19_full.npy"), test_list[0])
np.save(os.path.join(SAVE_DIR, "record20_full.npy"), test_list[1])

print(f"Saved in .\\{SAVE_DIR}\\")
print("Done.")