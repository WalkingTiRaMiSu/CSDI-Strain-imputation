import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

WINDOW_SIZE = 256

def make_block_mask_1d(length, missing_ratio=0.2, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    block_len = max(1, int(length * missing_ratio))
    if block_len >= length:
        block_len = length - 1

    start = rng.integers(0, length - block_len + 1)
    end = start + block_len

    mask = np.ones(length, dtype=np.float32)
    mask[start:end] = 0.0
    return mask

class TrainStrBlockDataset(Dataset):
    """
    train:
    - strain only
    - one window -> multiple masking scenarios
    """
    def __init__(
        self,
        train_windows,
        scenario_per_window=5,
        missing_ratio=0.2,
        seed=42,
    ):
        self.samples = []
        rng = np.random.default_rng(seed)

        for i in range(len(train_windows)):
            y = train_windows[i].astype(np.float32)  # (L,)
            for _ in range(scenario_per_window):
                gt_mask_1d = make_block_mask_1d(
                    len(y), missing_ratio=missing_ratio, rng=rng
                )
                observed_data = y[:, None]                # (L, 1)
                observed_mask = np.ones_like(observed_data, dtype=np.float32)
                gt_mask = gt_mask_1d[:, None].astype(np.float32)

                self.samples.append(
                    {
                        "observed_data": observed_data,
                        "observed_mask": observed_mask,
                        "gt_mask": gt_mask,
                        "timepoints": np.arange(len(y), dtype=np.float32),
                    }
                )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class ValidStrBlockDataset(Dataset):
    """
    validation:
    - train window 일부를 validation으로 사용
    - block masking 1개씩
    """
    def __init__(
        self,
        valid_windows,
        missing_ratio=0.2,
        seed=123,
    ):
        self.samples = []
        rng = np.random.default_rng(seed)

        for i in range(len(valid_windows)):
            y = valid_windows[i].astype(np.float32)
            gt_mask_1d = make_block_mask_1d(
                len(y), missing_ratio=missing_ratio, rng=rng
            )

            observed_data = y[:, None]
            observed_mask = np.ones_like(observed_data, dtype=np.float32)
            gt_mask = gt_mask_1d[:, None].astype(np.float32)

            self.samples.append(
                {
                    "observed_data": observed_data,
                    "observed_mask": observed_mask,
                    "gt_mask": gt_mask,
                    "timepoints": np.arange(len(y), dtype=np.float32),
                }
            )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class TestStrBlockDataset(Dataset):
    """
    test:
    - 19,20 windows
    - 각 window마다 큰 block mask 적용
    - window별 복원 후 평균내어 full signal 재구성
    """
    def __init__(
        self,
        test_windows,
        missing_ratio=0.2,
        seed=999,
    ):
        self.samples = []
        rng = np.random.default_rng(seed)

        for i in range(len(test_windows)):
            y = test_windows[i].astype(np.float32)
            gt_mask_1d = make_block_mask_1d(
                len(y), missing_ratio=missing_ratio, rng=rng
            )

            observed_data = y[:, None]
            observed_mask = np.ones_like(observed_data, dtype=np.float32)
            gt_mask = gt_mask_1d[:, None].astype(np.float32)

            self.samples.append(
                {
                    "observed_data": observed_data,
                    "observed_mask": observed_mask,
                    "gt_mask": gt_mask,
                    "timepoints": np.arange(len(y), dtype=np.float32),
                }
            )

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def get_dataloader(
    data_dir="./processed_str_05",
    batch_size=16,
    val_ratio=0.2,
    scenario_per_window=5,
    missing_ratio=0.2,
    seed=42,
):
    train_windows = np.load(os.path.join(data_dir, "train_windows.npy"))
    test_windows = np.load(os.path.join(data_dir, "test_windows.npy"))

    n_total = len(train_windows)
    idx = np.arange(n_total)

    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_valid = int(n_total * val_ratio)
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]

    train_base = train_windows[train_idx]
    valid_base = train_windows[valid_idx]

    train_dataset = TrainStrBlockDataset(
        train_base,
        scenario_per_window=scenario_per_window,
        missing_ratio=missing_ratio,
        seed=seed,
    )
    valid_dataset = ValidStrBlockDataset(
        valid_base,
        missing_ratio=missing_ratio,
        seed=seed + 100,
    )
    test_dataset = TestStrBlockDataset(
        test_windows,
        missing_ratio=missing_ratio,
        seed=seed + 200,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("=" * 60)
    print("Str-only block masking dataset")
    print("train base windows :", train_base.shape)
    print("valid base windows :", valid_base.shape)
    print("test  base windows :", test_windows.shape)
    print("train scenario data:", len(train_dataset))
    print("valid scenario data:", len(valid_dataset))
    print("test  scenario data:", len(test_dataset))
    print("=" * 60)

    return train_loader, valid_loader, test_loader