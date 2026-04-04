from scipy.io import loadmat
import numpy as np
import os

# =========================
# 설정
# =========================
MAT_FILE = "resp_total_re_05.mat"
WINDOW_SIZE = 256
STRIDE = 128
SAVE_DIR = "processed_05"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 1) 파일 읽기
# =========================
data = loadmat(MAT_FILE)
records = data["resp_total_re_05"]

print("=" * 60)
print(f"Loaded file: {MAT_FILE}")
print("records shape:", records.shape)
print("=" * 60)

all_X = []
all_Y = []

# =========================
# 2) 20개 record -> X, Y 만들기
# =========================
for i in range(20):
    rec = records[0, i]

    acc_g = np.asarray(rec["acc_g"]).squeeze()
    XTa   = np.asarray(rec["XTa"]).squeeze()
    XT    = np.asarray(rec["XT"]).squeeze()
    strr  = np.asarray(rec["str"]).squeeze()

    # 길이 확인
    T = len(acc_g)
    assert len(XTa) == T and len(XT) == T and len(strr) == T, f"Length mismatch in record {i+1}"

    X = np.column_stack([acc_g, XTa, XT]).astype(np.float32)   # (T, 3)
    Y = strr.astype(np.float32)                                # (T,)

    all_X.append(X)
    all_Y.append(Y)

    print(f"[Record {i+1:02d}] X shape = {X.shape}, Y shape = {Y.shape}")

# =========================
# 3) train / test 분리
# =========================
train_X_list = all_X[:18]   # 1~18
train_Y_list = all_Y[:18]

test_X_list = all_X[18:]    # 19~20
test_Y_list = all_Y[18:]

print("\n" + "=" * 60)
print("Split summary")
print(f"Train records: {len(train_X_list)} (1~18)")
print(f"Test  records: {len(test_X_list)} (19~20)")
print("=" * 60)

# =========================
# 4) normalization (train 기준)
#    X : min-max
#    Y : z-score
# =========================
# PDF의 ablation study에서 Half 방식:
# X는 min-max, Y는 z-score가 꽤 좋게 나옴
# 여기서는 일단 이 방식으로 진행
# =========================

# train X 전체 이어붙이기
train_X_concat = np.concatenate(train_X_list, axis=0)   # (sumT, 3)
train_Y_concat = np.concatenate(train_Y_list, axis=0)   # (sumT,)

x_min = train_X_concat.min(axis=0)   # (3,)
x_max = train_X_concat.max(axis=0)   # (3,)

y_mean = train_Y_concat.mean()
y_std  = train_Y_concat.std()

eps = 1e-8

def normalize_X(x):
    return (x - x_min) / (x_max - x_min + eps)

def normalize_Y(y):
    return (y - y_mean) / (y_std + eps)

train_X_list = [normalize_X(x).astype(np.float32) for x in train_X_list]
test_X_list  = [normalize_X(x).astype(np.float32) for x in test_X_list]

train_Y_list = [normalize_Y(y).astype(np.float32) for y in train_Y_list]
test_Y_list  = [normalize_Y(y).astype(np.float32) for y in test_Y_list]

print("\nNormalization info")
print("x_min :", x_min)
print("x_max :", x_max)
print("y_mean:", y_mean)
print("y_std :", y_std)

# =========================
# 5) sliding window
# =========================
def make_windows(X, Y, window_size=256, stride=128):
    X_windows = []
    Y_windows = []

    T = len(Y)
    start = 0

    while start + window_size <= T:
        end = start + window_size
        X_windows.append(X[start:end])   # (window_size, 3)
        Y_windows.append(Y[start:end])   # (window_size,)
        start += stride

    return X_windows, Y_windows

train_X_win = []
train_Y_win = []

test_X_win = []
test_Y_win = []

for i, (X, Y) in enumerate(zip(train_X_list, train_Y_list), start=1):
    xw, yw = make_windows(X, Y, WINDOW_SIZE, STRIDE)
    train_X_win.extend(xw)
    train_Y_win.extend(yw)
    print(f"Train record {i:02d} -> {len(xw)} windows")

for i, (X, Y) in enumerate(zip(test_X_list, test_Y_list), start=19):
    xw, yw = make_windows(X, Y, WINDOW_SIZE, STRIDE)
    test_X_win.extend(xw)
    test_Y_win.extend(yw)
    print(f"Test  record {i:02d} -> {len(xw)} windows")

train_X_win = np.array(train_X_win, dtype=np.float32)   # (N_train, 256, 3)
train_Y_win = np.array(train_Y_win, dtype=np.float32)   # (N_train, 256)

test_X_win  = np.array(test_X_win, dtype=np.float32)    # (N_test, 256, 3)
test_Y_win  = np.array(test_Y_win, dtype=np.float32)    # (N_test, 256)

print("\n" + "=" * 60)
print("Windowed dataset shape")
print("train_X_win:", train_X_win.shape)
print("train_Y_win:", train_Y_win.shape)
print("test_X_win :", test_X_win.shape)
print("test_Y_win :", test_Y_win.shape)
print("=" * 60)

# =========================
# 6) 저장
# =========================
np.save(os.path.join(SAVE_DIR, "train_X_win.npy"), train_X_win)
np.save(os.path.join(SAVE_DIR, "train_Y_win.npy"), train_Y_win)
np.save(os.path.join(SAVE_DIR, "test_X_win.npy"),  test_X_win)
np.save(os.path.join(SAVE_DIR, "test_Y_win.npy"),  test_Y_win)

# normalization 정보도 저장
np.savez(
    os.path.join(SAVE_DIR, "norm_info.npz"),
    x_min=x_min,
    x_max=x_max,
    y_mean=y_mean,
    y_std=y_std
)

print(f"\nSaved processed files in .\\{SAVE_DIR}\\")
print("Done.")