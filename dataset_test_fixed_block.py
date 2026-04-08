import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from fixed_mask_settings import (
    WINDOW_SIZE,
    STRIDE,
    MASK_19_START, MASK_19_END,
    MASK_20_START, MASK_20_END,
)

def make_global_block_mask(length, start, end):
    mask = np.ones(length, dtype=np.float32)
    start = max(0, start)
    end = min(length, end)
    mask[start:end] = 0.0
    return mask

def window_overlaps_missing(window_start, window_end, missing_start, missing_end):
    return not (window_end <= missing_start or window_start >= missing_end)

def build_windows_for_one_record(y_full, mask_full, missing_start, missing_end):
    """
    y_full: (T,)
    mask_full: (T,)
    return:
      windows_data: list of dict samples for CSDI
      starts: list of window start indices
    """
    samples = []
    starts = []

    T = len(y_full)
    start = 0
    while start + WINDOW_SIZE <= T:
        end = start + WINDOW_SIZE

        # 결측 구간과 겹치는 window만 사용
        if window_overlaps_missing(start, end, missing_start, missing_end):
            y_win = y_full[start:end].astype(np.float32)             # (L,)
            gt_mask_1d = mask_full[start:end].astype(np.float32)     # (L,)

            observed_data = y_win[:, None]                           # (L,1)
            observed_mask = np.ones_like(observed_data, dtype=np.float32)
            gt_mask = gt_mask_1d[:, None].astype(np.float32)

            samples.append(
                {
                    "observed_data": observed_data,
                    "observed_mask": observed_mask,
                    "gt_mask": gt_mask,
                    "timepoints": np.arange(len(y_win), dtype=np.float32),
                }
            )
            starts.append(start)

        start += STRIDE

    return samples, starts

class FixedBlockTestDataset(Dataset):
    def __init__(self, data_dir="./processed_str_05"):
        record19 = np.load(os.path.join(data_dir, "record19_full.npy"))  # normalized
        record20 = np.load(os.path.join(data_dir, "record20_full.npy"))  # normalized

        self.record19 = record19.astype(np.float32)
        self.record20 = record20.astype(np.float32)

        self.mask19 = make_global_block_mask(len(record19), MASK_19_START, MASK_19_END)
        self.mask20 = make_global_block_mask(len(record20), MASK_20_START, MASK_20_END)

        samples19, starts19 = build_windows_for_one_record(
            self.record19, self.mask19, MASK_19_START, MASK_19_END
        )
        samples20, starts20 = build_windows_for_one_record(
            self.record20, self.mask20, MASK_20_START, MASK_20_END
        )

        self.samples = samples19 + samples20
        self.starts19 = starts19
        self.starts20 = starts20

        self.n19 = len(samples19)
        self.n20 = len(samples20)

        print("=" * 60)
        print("Fixed-block test dataset")
        print(f"record19 length: {len(record19)}, missing: [{MASK_19_START}, {MASK_19_END})")
        print(f"record20 length: {len(record20)}, missing: [{MASK_20_START}, {MASK_20_END})")
        print(f"record19 windows used: {self.n19}")
        print(f"record20 windows used: {self.n20}")
        print(f"total test windows  : {len(self.samples)}")
        print("=" * 60)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

def get_test_dataloader(data_dir="./processed_str_05", batch_size=16):
    dataset = FixedBlockTestDataset(data_dir=data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset