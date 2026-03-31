import pandas as pd
import numpy as np
import os

# =========================
# 설정
# =========================
INPUT_CSV = "./custom_data/original.csv"
OUTPUT_CSV = "./custom_data/missing.csv"

SEED = 42
N_BLOCKS = 20          # 결측 구간 개수
MIN_BLOCK_LEN = 20     # 각 구간 최소 길이
MAX_BLOCK_LEN = 100    # 각 구간 최대 길이

np.random.seed(SEED)

# =========================
# 원본 불러오기
# =========================
df = pd.read_csv(INPUT_CSV)

# 컬럼 이름 확인
if "ch_8" not in df.columns:
    raise ValueError("original.csv 안에 'ch_8' 컬럼이 있어야 합니다.")

n = len(df)
missing_df = df.copy()

# 이미 결측 처리된 위치 기록용
used = np.zeros(n, dtype=bool)

blocks = []

# =========================
# 랜덤 block missing 만들기
# =========================
trial = 0
while len(blocks) < N_BLOCKS and trial < 10000:
    trial += 1

    block_len = np.random.randint(MIN_BLOCK_LEN, MAX_BLOCK_LEN + 1)
    start = np.random.randint(0, n - block_len)
    end = start + block_len

    # 겹치면 건너뜀
    if used[start:end].any():
        continue

    used[start:end] = True
    missing_df.loc[start:end-1, "ch_8"] = np.nan
    blocks.append((start, end - 1))

# =========================
# 저장
# =========================
missing_df.to_csv(OUTPUT_CSV, index=False)

print("missing.csv 생성 완료")
print(f"총 데이터 길이: {n}")
print(f"생성된 결측 구간 수: {len(blocks)}")
print("결측 구간 목록:")
for i, (s, e) in enumerate(blocks, 1):
    print(f"{i:02d}: {s} ~ {e}")