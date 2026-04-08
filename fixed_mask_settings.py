# =========================
# 여기만 바꾸면 됨
# =========================

# 학습된 모델이 들어있는 save 폴더 이름
TRAINED_SAVE_FOLDER_NAME = "str_block_05_20260407_101203"

# 이번 fixed-block 평가 결과를 저장할 새 폴더 이름
EVAL_SAVE_FOLDER_NAME = "str_block_eval_400_ns100_case1"

# 평가 시 nsample
NSAMPLE = 100   # 또는 100

# 샘플링 주파수
FS = 100.0  # Hz

# sliding window 설정 (학습 때와 동일해야 함)
WINDOW_SIZE = 256
STRIDE = 128

# =========================
# 19번, 20번 record의 결측 구간 (sample index 기준)
# 예: 8초~12초 결측이면 800~1200
# =========================
MASK_19_START = 800
MASK_19_END   = 1200

MASK_20_START = 2000
MASK_20_END   = 2600