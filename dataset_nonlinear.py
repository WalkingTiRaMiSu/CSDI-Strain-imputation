import os
import numpy as np
from torch.utils.data import Dataset, DataLoader


def build_full_data(x_win: np.ndarray, y_win: np.ndarray) -> np.ndarray:
    """
    x_win: (N, L, 3) = [acc_g, XTa, XT]
    y_win: (N, L)    = [str]
    return: (N, L, 4) = [acc_g, XTa, XT, str]
    """
    y_win = np.expand_dims(y_win, axis=-1)  # (N, L, 1)
    data = np.concatenate([x_win, y_win], axis=-1).astype(np.float32)
    return data


class NonlinearDataset(Dataset):
    """
    Vanilla CSDI-compatible dataset.
    observed_data : (L, K)
    observed_mask : (L, K)
    gt_mask       : (L, K)

    여기서는 baseline으로:
    - observed_data 에는 [acc_g, XTa, XT, str] 전체를 넣음
    - observed_mask 는 전부 1
    - gt_mask 는 test/valid에서 str 채널만 0으로 둬서
      [acc_g, XTa, XT] -> str 전체 예측이 되게 만듦

    train에서는 vanilla main_model의 random masking이 작동하므로
    gt_mask는 사실상 직접 학습에 쓰이지 않음.
    """

    def __init__(self, data_array: np.ndarray, use_index_list=None):
        self.data_array = data_array.astype(np.float32)

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.data_array))
        else:
            self.use_index_list = np.array(use_index_list)

        self.eval_length = self.data_array.shape[1]
        self.target_dim = self.data_array.shape[2]

    def __getitem__(self, org_index):
        idx = self.use_index_list[org_index]
        observed_data = self.data_array[idx].copy()  # (L, 4)

        observed_mask = np.ones_like(observed_data, dtype=np.float32)

        # baseline test pattern:
        # acc_g, XTa, XT는 조건으로 주고
        # str 전체를 복원 대상으로 둠
        gt_mask = np.ones_like(observed_data, dtype=np.float32)
        gt_mask[:, 3] = 0.0   # str channel 전체 missing target

        sample = {
            "observed_data": observed_data.astype(np.float32),
            "observed_mask": observed_mask.astype(np.float32),
            "gt_mask": gt_mask.astype(np.float32),
            "timepoints": np.arange(self.eval_length, dtype=np.float32),
        }
        return sample

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(
    seed=1,
    batch_size=16,
    val_ratio=0.2,
    data_dir="./processed_05",
):
    """
    processed_05 폴더에서 아래 파일을 읽음:
    - train_X_win.npy
    - train_Y_win.npy
    - test_X_win.npy
    - test_Y_win.npy
    """
    train_X = np.load(os.path.join(data_dir, "train_X_win.npy"))   # (1232, 256, 3)
    train_Y = np.load(os.path.join(data_dir, "train_Y_win.npy"))   # (1232, 256)
    test_X  = np.load(os.path.join(data_dir, "test_X_win.npy"))    # (67, 256, 3)
    test_Y  = np.load(os.path.join(data_dir, "test_Y_win.npy"))    # (67, 256)

    full_train = build_full_data(train_X, train_Y)  # (N, 256, 4)
    full_test  = build_full_data(test_X, test_Y)    # (M, 256, 4)

    n_total = len(full_train)
    all_idx = np.arange(n_total)

    rng = np.random.default_rng(seed)
    rng.shuffle(all_idx)

    n_valid = int(n_total * val_ratio)
    valid_idx = all_idx[:n_valid]
    train_idx = all_idx[n_valid:]

    train_dataset = NonlinearDataset(full_train, use_index_list=train_idx)
    valid_dataset = NonlinearDataset(full_train, use_index_list=valid_idx)
    test_dataset  = NonlinearDataset(full_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("=" * 60)
    print("Nonlinear dataset loaded")
    print("full_train shape:", full_train.shape)
    print("full_test  shape:", full_test.shape)
    print(f"train windows: {len(train_dataset)}")
    print(f"valid windows: {len(valid_dataset)}")
    print(f"test  windows: {len(test_dataset)}")
    print("=" * 60)

    return train_loader, valid_loader, test_loader