import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, original_csv, missing_csv):
        df_org = pd.read_csv(original_csv)
        df_mis = pd.read_csv(missing_csv)

        # 첫 번째 컬럼(ts) 제외
        value_cols = df_org.columns[1:]

        # 원본 / 결손 데이터
        org = df_org[value_cols].astype(float).values
        mis_raw = df_mis[value_cols].astype(float).values

        # mask 생성
        observed_mask = (~np.isnan(org)).astype(float)   # 원본에 값 있으면 1
        gt_mask = (~np.isnan(mis_raw)).astype(float)     # missing에 값 있으면 1, 결손이면 0

        # NaN -> 0
        org_filled = np.nan_to_num(org.copy())
        mis_filled = np.nan_to_num(mis_raw.copy())

        # 채널별 평균 / 표준편차 (원본 기준)
        mean = np.zeros(org.shape[1], dtype=np.float32)
        std = np.zeros(org.shape[1], dtype=np.float32)

        for k in range(org.shape[1]):
            valid = observed_mask[:, k] == 1
            if valid.sum() == 0:
                mean[k] = 0.0
                std[k] = 1.0
            else:
                mean[k] = org[valid, k].mean()
                std[k] = org[valid, k].std()
                if std[k] == 0 or np.isnan(std[k]):
                    std[k] = 1.0

        # 모델 입력은 missing 기준으로 생성
        observed_data = np.zeros_like(mis_filled, dtype=np.float32)
        for k in range(mis_filled.shape[1]):
            observed_data[:, k] = ((mis_filled[:, k] - mean[k]) / std[k]) * gt_mask[:, k]

        self.observed_values = observed_data.astype(np.float32)
        self.observed_masks = observed_mask.astype(np.float32)
        self.gt_masks = gt_mask.astype(np.float32)
        self.eval_length = len(org)

    def __getitem__(self, index):
        return {
            "observed_data": self.observed_values,
            "observed_mask": self.observed_masks,
            "gt_mask": self.gt_masks,
            "timepoints": np.arange(self.eval_length),
        }

    def __len__(self):
        return 1


def get_dataloader(batch_size=1):
    dataset = CustomDataset(
        "./custom_data/original.csv",
        "./custom_data/missing.csv",
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader