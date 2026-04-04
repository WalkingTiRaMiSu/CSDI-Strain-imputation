import numpy as np
import os

# =========================
# 설정
# =========================
LOAD_DIR = "processed_05"
SAVE_DIR = "csdi_input_05"
TRAIN_STR_MASK_RATIO = 0.2   # train에서 str의 20%를 랜덤 마스킹

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 1) 전처리된 데이터 불러오기
# =========================
train_X = np.load(os.path.join(LOAD_DIR, "train_X_win.npy"))   # (N, 256, 3)
train_Y = np.load(os.path.join(LOAD_DIR, "train_Y_win.npy"))   # (N, 256)

test_X  = np.load(os.path.join(LOAD_DIR, "test_X_win.npy"))    # (M, 256, 3)
test_Y  = np.load(os.path.join(LOAD_DIR, "test_Y_win.npy"))    # (M, 256)

print("=" * 60)
print("Loaded processed data")
print("train_X:", train_X.shape)
print("train_Y:", train_Y.shape)
print("test_X :", test_X.shape)
print("test_Y :", test_Y.shape)
print("=" * 60)

# =========================
# 2) feature 4개로 합치기
#    [acc_g, XTa, XT, str]
# =========================
train_Y_exp = np.expand_dims(train_Y, axis=-1)   # (N, 256, 1)
test_Y_exp  = np.expand_dims(test_Y, axis=-1)    # (M, 256, 1)

train_data = np.concatenate([train_X, train_Y_exp], axis=-1)   # (N, 256, 4)
test_data  = np.concatenate([test_X,  test_Y_exp], axis=-1)    # (M, 256, 4)

print("\nCombined data shape")
print("train_data:", train_data.shape)
print("test_data :", test_data.shape)

# =========================
# 3) 관측 마스크 만들기
#    feature index:
#    0 = acc_g
#    1 = XTa
#    2 = XT
#    3 = str
# =========================

# 기본은 전부 observed = 1
train_mask = np.ones_like(train_data, dtype=np.float32)
test_mask  = np.ones_like(test_data, dtype=np.float32)

# -------------------------
# train mask:
# str 채널(3번) 일부를 랜덤 마스킹
# -------------------------
rng = np.random.default_rng(seed=42)

N_train, T_train, F_train = train_data.shape

for i in range(N_train):
    num_mask = int(T_train * TRAIN_STR_MASK_RATIO)
    mask_idx = rng.choice(T_train, size=num_mask, replace=False)
    train_mask[i, mask_idx, 3] = 0.0

# -------------------------
# test mask:
# str 채널 전체를 마스킹
# -------------------------
test_mask[:, :, 3] = 0.0

print("\nMask shape")
print("train_mask:", train_mask.shape)
print("test_mask :", test_mask.shape)

# 마스킹 비율 확인
train_str_observed_ratio = train_mask[:, :, 3].mean()
test_str_observed_ratio  = test_mask[:, :, 3].mean()

print("\nObserved ratio of str channel")
print(f"train str observed ratio: {train_str_observed_ratio:.4f}")
print(f"test  str observed ratio: {test_str_observed_ratio:.4f}")

# =========================
# 4) masked input 만들기
#    mask=0인 곳은 0으로 가림
# =========================
train_masked_data = train_data * train_mask
test_masked_data  = test_data * test_mask

print("\nMasked data shape")
print("train_masked_data:", train_masked_data.shape)
print("test_masked_data :", test_masked_data.shape)

# =========================
# 5) 저장
# =========================
np.save(os.path.join(SAVE_DIR, "train_data.npy"), train_data)
np.save(os.path.join(SAVE_DIR, "test_data.npy"), test_data)

np.save(os.path.join(SAVE_DIR, "train_mask.npy"), train_mask)
np.save(os.path.join(SAVE_DIR, "test_mask.npy"), test_mask)

np.save(os.path.join(SAVE_DIR, "train_masked_data.npy"), train_masked_data)
np.save(os.path.join(SAVE_DIR, "test_masked_data.npy"), test_masked_data)

# 간단한 메타 정보 저장
np.savez(
    os.path.join(SAVE_DIR, "meta_info.npz"),
    train_str_mask_ratio=TRAIN_STR_MASK_RATIO
)

print(f"\nSaved CSDI input files in .\\{SAVE_DIR}\\")
print("Done.")