import pandas as pd
import numpy as np

# =========================
# 경로 설정
# =========================
INPUT_CSV = "./custom_data/original.csv"
OUTPUT_CSV = "./custom_data/missing.csv"

# =========================
# 랜덤 결측 설정
# =========================
SEED = 42
N_BLOCKS = 20          # 결측 블록 개수
MIN_BLOCK_LEN = 20     # 블록 최소 길이
MAX_BLOCK_LEN = 100    # 블록 최대 길이

TARGET_COL = "ch_8"    # 결측 만들 채널

np.random.seed(SEED)

# =========================
# 원본 읽기
# =========================
df = pd.read_csv(INPUT_CSV)

if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

missing_df = df.copy()
n = len(df)

# 겹치지 않게 만들기 위한 mask
used = np.zeros(n, dtype=bool)
blocks = []

trial = 0
while len(blocks) < N_BLOCKS and trial < 10000:
    trial += 1

    block_len = np.random.randint(MIN_BLOCK_LEN, MAX_BLOCK_LEN + 1)
    start = np.random.randint(0, n - block_len + 1)
    end = start + block_len

    # 이미 결측으로 잡은 구간과 겹치면 다시 뽑기
    if used[start:end].any():
        continue

    used[start:end] = True

    # ch_8만 NaN으로 만들기
    missing_df.loc[start:end-1, TARGET_COL] = np.nan
    blocks.append((start, end - 1))

# =========================
# 저장
# =========================
missing_df.to_csv(OUTPUT_CSV, index=False)

print("missing.csv 생성 완료")
print(f"총 데이터 길이: {n}")
print(f"생성된 결측 블록 수: {len(blocks)}")
print("결측 구간 목록:")
for i, (s, e) in enumerate(blocks, 1):
    print(f"{i:02d}: {s} ~ {e}")