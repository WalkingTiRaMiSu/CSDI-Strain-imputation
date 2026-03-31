import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


def get_feature_columns(df: pd.DataFrame):
    # ch_1 ~ ch_8만 feature로 사용
    feature_cols = [c for c in df.columns if str(c).startswith("ch_")]
    if len(feature_cols) == 0:
        raise ValueError(f"ch_ 로 시작하는 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")
    return feature_cols


def compute_mean_std(values: np.ndarray):
    n_feat = values.shape[1]
    mean = np.zeros(n_feat, dtype=np.float32)
    std = np.zeros(n_feat, dtype=np.float32)

    for k in range(n_feat):
        col = values[:, k]
        valid = ~np.isnan(col)
        if valid.sum() == 0:
            mean[k] = 0.0
            std[k] = 1.0
        else:
            mean[k] = col[valid].mean()
            std[k] = col[valid].std()
            if std[k] == 0 or np.isnan(std[k]):
                std[k] = 1.0
    return mean, std


def normalize(values: np.ndarray, mean: np.ndarray, std: np.ndarray):
    out = np.zeros_like(values, dtype=np.float32)
    for k in range(values.shape[1]):
        out[:, k] = (values[:, k] - mean[k]) / std[k]
    return out.astype(np.float32)


def make_ch8_block_mask(window_len, n_feat, ch8_index, missing_ratio, seed):
    """
    모든 채널은 observed(1), ch8만 block missing(0) 만들기
    """
    rng = np.random.default_rng(seed)

    block_len = max(1, int(window_len * missing_ratio))
    if block_len >= window_len:
        block_len = window_len - 1

    start = rng.integers(0, window_len - block_len + 1)
    end = start + block_len

    mask = np.ones((window_len, n_feat), dtype=np.float32)
    mask[start:end, ch8_index] = 0.0
    return mask


class MultiChannelTrainDataset(Dataset):
    """
    학습/검증용:
    original.csv만 사용
    ch1~7은 항상 관측
    ch8만 랜덤 block missing을 만들어 복원 학습
    """

    def __init__(self, original_csv, start_list, window_len, missing_ratio, seed):
        df = pd.read_csv(original_csv)
        feature_cols = get_feature_columns(df)
        org = df[feature_cols].astype(float).values

        self.feature_cols = feature_cols
        self.mean, self.std = compute_mean_std(org)
        org_norm = normalize(org, self.mean, self.std)

        if "ch_8" not in feature_cols:
            raise ValueError(f"feature 컬럼 안에 ch_8이 없습니다. 현재 feature_cols={feature_cols}")
        ch8_index = feature_cols.index("ch_8")

        self.samples = []

        for i, start in enumerate(start_list):
            end = start + window_len
            window = org_norm[start:end].astype(np.float32)

            # original은 온전하므로 observed_mask는 전부 1
            observed_mask = np.ones_like(window, dtype=np.float32)

            # ch8만 랜덤 block missing
            gt_mask = make_ch8_block_mask(
                window_len=window_len,
                n_feat=window.shape[1],
                ch8_index=ch8_index,
                missing_ratio=missing_ratio,
                seed=seed + i,
            )

            self.samples.append(
                {
                    "observed_data": window,                         # 전체 normalized 원본
                    "observed_mask": observed_mask,                  # 전체 관측됨
                    "gt_mask": gt_mask.astype(np.float32),           # ch8 block만 0
                    "timepoints": np.arange(window_len, dtype=np.float32),
                }
            )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class MultiChannelTestDataset(Dataset):
    """
    테스트용:
    original.csv + missing.csv 사용
    ch1~7은 full observed
    ch8은 missing.csv의 NaN 구간만 복원
    """

    def __init__(self, original_csv, missing_csv):
        df_org = pd.read_csv(original_csv)
        df_mis = pd.read_csv(missing_csv)

        feature_cols = get_feature_columns(df_org)
        org = df_org[feature_cols].astype(float).values
        mis = df_mis[feature_cols].astype(float).values

        self.feature_cols = feature_cols

        mean, std = compute_mean_std(org)
        org_norm = normalize(org, mean, std)

        # missing.csv 기준: 값 있으면 1, NaN이면 0
        gt_mask = (~np.isnan(mis)).astype(np.float32)

        # 모델 입력에는 full normalized data를 넣고
        # cond_mask(gt_mask)로 어느 위치를 조건/복원으로 쓸지 구분
        observed_data = org_norm.astype(np.float32)

        observed_mask = np.ones_like(observed_data, dtype=np.float32)

        self.sample = {
            "observed_data": observed_data,
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(len(org_norm), dtype=np.float32),
        }

    def __getitem__(self, index):
        return self.sample

    def __len__(self):
        return 1


def get_dataloader(
    batch_size=1,
    window_len=128,
    stride=16,
    train_missing_ratio=0.2,
    seed=42,
):
    original_csv = "./custom_data/original.csv"
    missing_csv = "./custom_data/missing.csv"

    df_org = pd.read_csv(original_csv)
    feature_cols = get_feature_columns(df_org)
    org = df_org[feature_cols].astype(float).values

    total_len = len(org)
    start_list = list(range(0, total_len - window_len + 1, stride))

    if len(start_list) < 10:
        raise ValueError("윈도우 개수가 너무 적습니다. window_len 또는 stride를 조정하세요.")

    n_total = len(start_list)
    n_train = int(n_total * 0.7)
    n_valid = int(n_total * 0.15)

    train_starts = start_list[:n_train]
    valid_starts = start_list[n_train:n_train + n_valid]

    train_dataset = MultiChannelTrainDataset(
        original_csv=original_csv,
        start_list=train_starts,
        window_len=window_len,
        missing_ratio=train_missing_ratio,
        seed=seed,
    )

    valid_dataset = MultiChannelTrainDataset(
        original_csv=original_csv,
        start_list=valid_starts,
        window_len=window_len,
        missing_ratio=train_missing_ratio,
        seed=seed + 1000,
    )

    test_dataset = MultiChannelTestDataset(
        original_csv=original_csv,
        missing_csv=missing_csv,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    return train_loader, valid_loader, test_loader