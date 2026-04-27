# dataset_strain_global.py
# Global STr block missing 전용 dataset / preprocessing.
# 핵심: 먼저 절대 시간축 기준 STr 결측 block을 정하고, 이후 sliding window를 만든다.

import os
import glob
import json
import csv
import math
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# MAT loading
# -----------------------------

def find_mat_file(mat_path=None):
    candidates = []
    if mat_path:
        candidates.append(mat_path)
        if not str(mat_path).lower().endswith(".mat"):
            candidates.append(str(mat_path) + ".mat")

    candidates += [
        "resp_total_re_05.mat",
        "resp_total_re_05(9).mat",
        "resp_total_re_05",
    ]
    candidates += glob.glob("resp_total*.mat")
    candidates += glob.glob("resp_total*")

    seen = []
    for c in candidates:
        if c not in seen:
            seen.append(c)

    for c in seen:
        if os.path.isfile(c):
            return c

    print("[ERROR] 현재 폴더 파일 목록:")
    for f in os.listdir("."):
        print(" -", f)
    raise FileNotFoundError("resp_total mat 파일을 찾지 못했습니다. 현재 폴더에 resp_total_re_05.mat를 두세요.")


def _get_field(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"MAT struct에서 field를 찾지 못했습니다: {names}")


def list_available_fields(item):
    fields = []
    if hasattr(item, "_fieldnames") and item._fieldnames is not None:
        fields = list(item._fieldnames)
    else:
        fields = [k for k in dir(item) if not k.startswith("_")]
    return fields


def load_resp_total(mat_path=None):
    mat_path = find_mat_file(mat_path)
    print(f"[INFO] MAT file: {mat_path}")

    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    print(f"[INFO] MAT keys: {keys}")

    if "resp_total_re_05" in mat:
        key = "resp_total_re_05"
    elif len(keys) == 1:
        key = keys[0]
    else:
        matched = [k for k in keys if k.startswith("resp_total")]
        if len(matched) == 1:
            key = matched[0]
        else:
            raise KeyError("MAT 내부 변수명을 자동으로 고르지 못했습니다. resp_total_re_05인지 확인하세요.")

    resp_total = np.ravel(mat[key])
    events = []

    for i, item in enumerate(resp_total):
        if i == 0:
            print(f"[INFO] available fields in event struct: {list_available_fields(item)}")

        t = np.asarray(_get_field(item, ["t", "time", "Time"])).squeeze().astype(float)
        xta = np.asarray(_get_field(item, ["XTa", "xta", "XTA"])).squeeze().astype(float)
        str_data = np.asarray(_get_field(item, ["str", "STr", "Str", "strain", "Strain"])).squeeze().astype(float)

        n = int(min(len(np.ravel(xta)), len(np.ravel(str_data))))
        xta = np.ravel(xta)[:n]
        str_data = np.ravel(str_data)[:n]
        t = np.ravel(t)
        if len(t) != n:
            t = np.arange(n, dtype=float)
        else:
            t = t[:n]

        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        else:
            dt = 1.0

        events.append({
            "event_no": i + 1,
            "t": t.astype(float),
            "XTa": xta.astype(float),
            "STr": str_data.astype(float),
            "dt": dt,
            "length": n,
        })

    print(f"[INFO] loaded events: {len(events)}")
    return events


# -----------------------------
# Feature / scaling / active interval
# -----------------------------

def make_features(event, use_derivatives=True):
    xta = event["XTa"].astype(float)
    str_data = event["STr"].astype(float)
    dt = float(event.get("dt", 1.0))

    if use_derivatives:
        dxta = np.gradient(xta, dt)
        ddxta = np.gradient(dxta, dt)
        data = np.stack([xta, dxta, ddxta, str_data], axis=1)
        feature_names = ["XTa", "dXTa", "ddXTa", "STr"]
        str_index = 3
    else:
        data = np.stack([xta, str_data], axis=1)
        feature_names = ["XTa", "STr"]
        str_index = 1

    return data.astype(np.float32), feature_names, str_index


def round_up_to_multiple(x, base):
    base = int(base)
    if base <= 1:
        return int(math.ceil(float(x)))
    return int(math.ceil(float(x) / base) * base)


def clip_interval(start, length, n):
    length = int(min(max(1, length), n))
    start = int(round(start))
    end = start + length
    if start < 0:
        start = 0
        end = length
    if end > n:
        end = n
        start = n - length
    return int(start), int(end)


def active_interval_by_xta(event, data_cfg):
    n = int(event["length"])
    xta = event["XTa"]
    peak_idx = int(np.argmax(np.abs(xta)))

    active_ratio = float(data_cfg.get("active_ratio", 0.30))
    active_min = int(data_cfg.get("active_min", 1500))
    active_max = int(data_cfg.get("active_max", 3000))
    active_round_to = int(data_cfg.get("active_round_to", 500))
    left_fraction = float(data_cfg.get("active_left_fraction", 0.40))

    raw_len = active_ratio * n
    active_len = round_up_to_multiple(raw_len, active_round_to)
    active_len = max(active_min, active_len)
    active_len = min(active_max, active_len)
    active_len = min(active_len, n)

    start = peak_idx - int(round(left_fraction * active_len))
    active_start, active_end = clip_interval(start, active_len, n)
    return int(active_start), int(active_end), int(peak_idx), int(active_len)


def compute_scaler(events, event_nos, data_cfg, use_derivatives):
    vals = []
    feature_names = None
    str_index = None
    for event_no in event_nos:
        event = events[event_no - 1]
        features, feature_names, str_index = make_features(event, use_derivatives)
        active_start, active_end, _, _ = active_interval_by_xta(event, data_cfg)
        vals.append(features[active_start:active_end])

    vals = np.concatenate(vals, axis=0)
    mean = vals.mean(axis=0).astype(np.float32)
    std = vals.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std, feature_names, str_index


# -----------------------------
# Global missing scenario generation
# -----------------------------

def missing_len_from_ratio(active_len, ratio, data_cfg):
    missing_round_to = int(data_cfg.get("missing_round_to", 50))
    missing_min = int(data_cfg.get("missing_min", 100))
    missing_max = int(data_cfg.get("missing_max", 500))

    m = round_up_to_multiple(float(active_len) * float(ratio), missing_round_to)
    m = max(missing_min, m)
    m = min(missing_max, m)
    m = min(m, active_len)
    return int(max(1, m))


def global_block_for_position(active_start, active_end, peak_idx, missing_len, position):
    active_len = int(active_end - active_start)
    max_start = int(active_end - missing_len)

    if position == "peak_center":
        start = int(peak_idx - missing_len // 2)
    elif position == "peak_before":
        start = int(peak_idx - missing_len)
    elif position == "peak_after":
        start = int(peak_idx)
    elif position == "active_left":
        start = int(active_start + round(0.15 * max(active_len - missing_len, 0)))
    elif position == "active_middle":
        start = int(active_start + round(0.50 * max(active_len - missing_len, 0)))
    elif position == "active_right":
        start = int(active_start + round(0.85 * max(active_len - missing_len, 0)))
    else:
        raise ValueError(f"지원하지 않는 global missing position: {position}")

    start = int(np.clip(start, active_start, max_start))
    end = int(start + missing_len)
    return start, end


def window_starts_for_active(active_start, active_end, window_size, stride):
    active_len = int(active_end - active_start)
    if active_len < window_size:
        raise ValueError(
            f"active interval 길이({active_len})가 window_size({window_size})보다 작습니다. "
            f"window_size를 줄이거나 active_min을 키우세요."
        )

    last_start = int(active_end - window_size)
    starts = list(range(int(active_start), last_start + 1, int(stride)))
    if len(starts) == 0 or starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_global_samples(
    events,
    event_nos,
    data_cfg,
    use_derivatives=True,
    split_name="train",
    seed=1,
):
    rng = np.random.default_rng(seed)
    window_size = int(data_cfg.get("window_size", 1000))
    stride = int(data_cfg.get("stride", 250))
    min_target_points = int(data_cfg.get("min_target_points", 1))

    if split_name == "test":
        ratios = data_cfg.get("test_missing_ratios", data_cfg.get("missing_ratios", [0.05, 0.10, 0.15]))
        positions = data_cfg.get("test_missing_positions", ["peak_center"])
        jitter_repeats = 1
        jitter_fraction = 0.0
    else:
        ratios = data_cfg.get("missing_ratios", [0.05, 0.10, 0.15])
        positions = data_cfg.get("train_missing_positions", ["peak_before", "peak_center", "peak_after"])
        jitter_repeats = int(data_cfg.get("train_jitter_repeats", 1))
        jitter_fraction = float(data_cfg.get("train_jitter_fraction_of_missing", 0.0))
        jitter_repeats = max(1, jitter_repeats)

    samples = []
    scenarios = []
    feature_names = None
    str_index = None

    for event_no in event_nos:
        event = events[event_no - 1]
        _, feature_names, str_index = make_features(event, use_derivatives)
        active_start, active_end, peak_idx, active_len = active_interval_by_xta(event, data_cfg)
        starts = window_starts_for_active(active_start, active_end, window_size, stride)

        for ratio in ratios:
            missing_len = missing_len_from_ratio(active_len, float(ratio), data_cfg)
            for position in positions:
                base_m0, base_m1 = global_block_for_position(
                    active_start, active_end, peak_idx, missing_len, position
                )
                for rep in range(jitter_repeats):
                    if rep == 0 or jitter_fraction <= 0:
                        m0, m1 = base_m0, base_m1
                        jitter = 0
                    else:
                        max_jitter = int(round(jitter_fraction * missing_len))
                        jitter = int(rng.integers(-max_jitter, max_jitter + 1)) if max_jitter > 0 else 0
                        m0 = int(np.clip(base_m0 + jitter, active_start, active_end - missing_len))
                        m1 = int(m0 + missing_len)

                    scenario_key = (
                        f"EQ{event_no:02d}_{split_name}_ratio{float(ratio):.2f}_"
                        f"{position}_rep{rep:02d}"
                    )
                    scenario = {
                        "scenario_key": scenario_key,
                        "event_no": int(event_no),
                        "split": split_name,
                        "position": position,
                        "missing_ratio": float(ratio),
                        "jitter_repeat": int(rep),
                        "jitter_points": int(jitter),
                        "active_start": int(active_start),
                        "active_end": int(active_end),
                        "active_len": int(active_len),
                        "xta_peak_idx": int(peak_idx),
                        "missing_start": int(m0),
                        "missing_end": int(m1),
                        "missing_len": int(m1 - m0),
                    }
                    scenarios.append(scenario)

                    window_id = 0
                    for s in starts:
                        e = int(s + window_size)
                        inter0 = max(int(s), int(m0))
                        inter1 = min(int(e), int(m1))
                        target_points = int(max(0, inter1 - inter0))
                        if target_points < min_target_points:
                            continue

                        samples.append({
                            **scenario,
                            "window_id": int(window_id),
                            "start_idx": int(s),
                            "end_idx": int(e),
                            "target_start_idx": int(inter0),
                            "target_end_idx": int(inter1),
                            "target_points": int(target_points),
                        })
                        window_id += 1

    return samples, scenarios, feature_names, str_index


def split_samples_by_scenario(samples, valid_ratio=0.2, seed=1):
    if len(samples) == 0:
        return [], []
    scenario_keys = sorted(set([s["scenario_key"] for s in samples]))
    rng = np.random.default_rng(seed)
    order = np.arange(len(scenario_keys))
    rng.shuffle(order)

    n_valid = max(1, int(round(len(order) * float(valid_ratio)))) if len(order) > 1 else 0
    valid_keys = set([scenario_keys[i] for i in order[:n_valid]])

    train_samples = [s for s in samples if s["scenario_key"] not in valid_keys]
    valid_samples = [s for s in samples if s["scenario_key"] in valid_keys]
    return train_samples, valid_samples


class StrainGlobalMaskDataset(Dataset):
    def __init__(
        self,
        events,
        samples,
        mean,
        std,
        feature_names,
        str_index,
        window_size,
        use_derivatives=True,
        split_name="train",
    ):
        self.events = events
        self.samples = list(samples)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        self.feature_names = list(feature_names)
        self.str_index = int(str_index)
        self.window_size = int(window_size)
        self.use_derivatives = bool(use_derivatives)
        self.split_name = split_name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        event = self.events[item["event_no"] - 1]
        features, _, _ = make_features(event, self.use_derivatives)

        s = int(item["start_idx"])
        e = int(item["end_idx"])
        values = features[s:e]
        values_norm = (values - self.mean) / self.std

        observed_mask = np.ones_like(values_norm, dtype=np.float32)
        gt_mask = np.ones_like(values_norm, dtype=np.float32)

        # Global STr missing block과 현재 window의 교집합만 STr에서 mask 처리
        local0 = int(item["target_start_idx"] - s)
        local1 = int(item["target_end_idx"] - s)
        gt_mask[local0:local1, self.str_index] = 0.0

        return {
            "observed_data": values_norm.astype(np.float32),
            "observed_mask": observed_mask,
            "gt_mask": gt_mask,
            "timepoints": np.arange(self.window_size, dtype=np.float32),
            "event_no": np.int64(item["event_no"]),
            "window_id": np.int64(item["window_id"]),
            "start_idx": np.int64(item["start_idx"]),
            "end_idx": np.int64(item["end_idx"]),
            "active_start": np.int64(item["active_start"]),
            "active_end": np.int64(item["active_end"]),
            "xta_peak_idx": np.int64(item["xta_peak_idx"]),
            "missing_start": np.int64(item["missing_start"]),
            "missing_end": np.int64(item["missing_end"]),
            "target_start_idx": np.int64(item["target_start_idx"]),
            "target_end_idx": np.int64(item["target_end_idx"]),
            "missing_ratio": np.float32(item["missing_ratio"]),
            "scenario_index": np.int64(idx),
        }


# -----------------------------
# Summary save / dataloader
# -----------------------------

def save_scenario_csv(path, scenarios):
    if len(scenarios) == 0:
        return
    keys = [
        "scenario_key", "event_no", "split", "position", "missing_ratio",
        "jitter_repeat", "jitter_points", "active_start", "active_end", "active_len",
        "xta_peak_idx", "missing_start", "missing_end", "missing_len"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in scenarios:
            writer.writerow({k: row.get(k, "") for k in keys})


def save_dataset_summary(path, events, info):
    rows = []
    data_cfg = info["data_config"]
    for event in events:
        active_start, active_end, xta_peak_idx, active_len = active_interval_by_xta(event, data_cfg)
        str_peak_idx = int(np.argmax(np.abs(event["STr"])))
        rows.append({
            "event_no": int(event["event_no"]),
            "length": int(event["length"]),
            "t_start": float(event["t"][0]),
            "t_end": float(event["t"][-1]),
            "dt": float(event["dt"]),
            "xta_peak_idx": int(xta_peak_idx),
            "xta_peak_time": float(event["t"][xta_peak_idx]),
            "xta_peak_value": float(event["XTa"][xta_peak_idx]),
            "str_peak_idx_reference_only": int(str_peak_idx),
            "str_peak_time_reference_only": float(event["t"][str_peak_idx]),
            "active_start": int(active_start),
            "active_end": int(active_end),
            "active_len": int(active_len),
        })

    summary = {
        "events": rows,
        "info": info,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def _scenario_count(samples):
    return len(set([s["scenario_key"] for s in samples]))


def get_dataloader(config, seed=1):
    data_cfg = config["data"]
    events = load_resp_total(data_cfg.get("mat_path", None))

    train_events = data_cfg.get("train_events", list(range(1, 19)))
    test_events = data_cfg.get("test_events", [19, 20])
    use_derivatives = bool(data_cfg.get("use_derivatives", True))
    window_size = int(data_cfg.get("window_size", 1000))
    valid_ratio = float(data_cfg.get("valid_ratio", 0.2))

    mean, std, feature_names, str_index = compute_scaler(
        events, train_events, data_cfg, use_derivatives
    )

    all_train_samples, train_scenarios, feature_names, str_index = build_global_samples(
        events, train_events, data_cfg, use_derivatives=use_derivatives, split_name="train", seed=seed
    )
    train_samples, valid_samples = split_samples_by_scenario(
        all_train_samples, valid_ratio=valid_ratio, seed=seed
    )
    valid_scenario_keys = set([s["scenario_key"] for s in valid_samples])
    train_scenarios_used = [s for s in train_scenarios if s["scenario_key"] not in valid_scenario_keys]
    valid_scenarios_used = [s for s in train_scenarios if s["scenario_key"] in valid_scenario_keys]

    test_samples, test_scenarios, _, _ = build_global_samples(
        events, test_events, data_cfg, use_derivatives=use_derivatives, split_name="test", seed=seed + 2000
    )

    train_dataset = StrainGlobalMaskDataset(
        events, train_samples, mean, std, feature_names, str_index,
        window_size=window_size, use_derivatives=use_derivatives, split_name="train"
    )
    valid_dataset = StrainGlobalMaskDataset(
        events, valid_samples, mean, std, feature_names, str_index,
        window_size=window_size, use_derivatives=use_derivatives, split_name="valid"
    )
    test_dataset = StrainGlobalMaskDataset(
        events, test_samples, mean, std, feature_names, str_index,
        window_size=window_size, use_derivatives=use_derivatives, split_name="test"
    )

    batch_size = int(config["train"].get("batch_size", 16))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # test reconstruction ratio per scenario
    test_recon_summary = []
    for sc in test_scenarios:
        event = events[sc["event_no"] - 1]
        n = int(event["length"])
        active_len = int(sc["active_end"] - sc["active_start"])
        miss_len = int(sc["missing_end"] - sc["missing_start"])
        test_recon_summary.append({
            "scenario_key": sc["scenario_key"],
            "event_no": int(sc["event_no"]),
            "position": sc["position"],
            "missing_ratio": float(sc["missing_ratio"]),
            "length": int(n),
            "active_start": int(sc["active_start"]),
            "active_end": int(sc["active_end"]),
            "active_ratio_of_full": float(active_len / n),
            "missing_start": int(sc["missing_start"]),
            "missing_end": int(sc["missing_end"]),
            "missing_points": int(miss_len),
            "missing_ratio_of_full": float(miss_len / n),
            "missing_ratio_of_active": float(miss_len / max(active_len, 1)),
        })

    info = {
        "feature_names": feature_names,
        "str_index": int(str_index),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "normalization": "Per-feature z-score. Mean/std are computed only from train events' XTa-peak-based active intervals. Test event statistics are not used.",
        "train_events": train_events,
        "test_events": test_events,
        "data_config": data_cfg,
        "train_scenarios": _scenario_count(train_samples),
        "valid_scenarios": _scenario_count(valid_samples),
        "test_scenarios": _scenario_count(test_samples),
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "test_samples": len(test_dataset),
        "scenario_counts": {
            "all_train_before_split": len(set([s["scenario_key"] for s in all_train_samples])),
            "train_after_split": _scenario_count(train_samples),
            "valid_after_split": _scenario_count(valid_samples),
            "test": _scenario_count(test_samples),
        },
        "test_recon_summary": test_recon_summary,
    }

    print("[INFO] feature_names:", feature_names)
    print("[INFO] str_index:", str_index)
    print("[INFO] normalization: train active intervals only, per-feature z-score")
    print("[INFO] scaler mean:", [round(float(x), 6) for x in mean])
    print("[INFO] scaler std :", [round(float(x), 6) for x in std])
    print(f"[INFO] scenarios train/valid/test: {_scenario_count(train_samples)} / {_scenario_count(valid_samples)} / {_scenario_count(test_samples)}")
    print(f"[INFO] samples   train/valid/test: {len(train_dataset)} / {len(valid_dataset)} / {len(test_dataset)}")
    print("[INFO] test global missing scenarios:")
    for row in test_recon_summary:
        print(
            f"  EQ {row['event_no']:02d} | {row['position']:<12} | ratio={row['missing_ratio']:.2f} | "
            f"missing={row['missing_points']} pt "
            f"({100*row['missing_ratio_of_full']:.2f}% full, {100*row['missing_ratio_of_active']:.2f}% active)"
        )

    # scenario tables are attached to datasets for exe to save after folder creation
    train_dataset.scenarios = train_scenarios_used
    valid_dataset.scenarios = valid_scenarios_used
    test_dataset.scenarios = test_scenarios

    return train_loader, valid_loader, test_loader, train_dataset, valid_dataset, test_dataset, info
