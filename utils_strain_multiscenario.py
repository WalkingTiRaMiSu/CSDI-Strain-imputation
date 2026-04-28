# utils_strain_multiscenario.py
# Training / evaluation / plotting for multi-scenario STr CSDI.

import os
import csv
import pickle
import time
import re
import json
import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.switch_backend("Agg")


def apply_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 18


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", str(text))


def save_loss_history(foldername, train_losses, valid_losses):
    path = os.path.join(foldername, "loss_history.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "valid_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, valid_losses), start=1):
            writer.writerow([i, tr, va])


def plot_loss_curve(foldername, train_losses, valid_losses):
    apply_plot_style()
    path = os.path.join(foldername, "loss_curve.png")
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, train_losses, linewidth=1.8, label="Train loss")

    valid_epochs = []
    valid_vals = []
    for i, v in enumerate(valid_losses, start=1):
        if not np.isnan(v):
            valid_epochs.append(i)
            valid_vals.append(v)
    if len(valid_vals) > 0:
        plt.plot(valid_epochs, valid_vals, marker="o", linewidth=1.8, label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def train(model, config, train_loader, valid_loader=None, valid_epoch_interval=10, foldername=""):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config.get("weight_decay", 1e-6)))

    if foldername != "":
        os.makedirs(foldername, exist_ok=True)
        output_path = os.path.join(foldername, "model.pth")

    epochs = int(config["epochs"])
    p1 = int(0.75 * epochs)
    p2 = int(0.9 * epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)

    itr_per_epoch = int(float(config.get("itr_per_epoch", 1e8)))
    gradient_clip = config.get("gradient_clip", None)
    gradient_clip = None if gradient_clip is None else float(gradient_clip)

    train_losses = []
    valid_losses = []
    best_valid_loss = 1e10

    train_batches_total = len(train_loader)
    valid_batches_total = len(valid_loader) if valid_loader is not None else 0
    effective_train_batches = min(train_batches_total, itr_per_epoch)

    print("\n" + "=" * 90)
    print("[TRAIN SETUP] CSDI strain MULTI-SCENARIO training")
    print(f"[TRAIN SETUP] epochs              : {epochs}")
    print(f"[TRAIN SETUP] batch_size          : {config.get('batch_size', 'unknown')}")
    print(f"[TRAIN SETUP] learning_rate       : {float(config['lr']):.6g}")
    print(f"[TRAIN SETUP] train batches/epoch : {effective_train_batches} / {train_batches_total}")
    print(f"[TRAIN SETUP] valid batches       : {valid_batches_total}")
    print(f"[TRAIN SETUP] valid interval      : every {valid_epoch_interval} epoch(s)")
    print(f"[TRAIN SETUP] gradient_clip       : {gradient_clip}")
    print("=" * 90 + "\n")

    total_start_time = time.time()

    for epoch_no in range(epochs):
        epoch = epoch_no + 1
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch:03d}/{epochs:03d}] START | lr={current_lr:.6g}")

        model.train()
        avg_loss = 0.0
        batch_count = 0

        with tqdm(train_loader, desc=f"Train {epoch:03d}/{epochs:03d}", mininterval=1.0, maxinterval=10.0, dynamic_ncols=True) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                if gradient_clip is not None and gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()

                avg_loss += loss.item()
                batch_count = batch_no
                running = avg_loss / batch_no
                it.set_postfix(ordered_dict={"loss": f"{loss.item():.5f}", "avg": f"{running:.5f}", "lr": f"{current_lr:.2e}"}, refresh=True)

                if batch_no >= itr_per_epoch:
                    break

        lr_scheduler.step()
        train_loss = avg_loss / max(batch_count, 1)
        train_losses.append(train_loss)

        valid_loss = np.nan
        did_validation = valid_loader is not None and epoch % int(valid_epoch_interval) == 0

        if did_validation:
            print(f"[Epoch {epoch:03d}/{epochs:03d}] VALIDATION START")
            model.eval()
            avg_loss_valid = 0.0
            valid_count = 0
            with torch.no_grad():
                with tqdm(valid_loader, desc=f"Valid {epoch:03d}/{epochs:03d}", mininterval=1.0, maxinterval=10.0, dynamic_ncols=True) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        valid_count = batch_no
                        running_valid = avg_loss_valid / batch_no
                        it.set_postfix(ordered_dict={"valid_loss": f"{loss.item():.5f}", "valid_avg": f"{running_valid:.5f}"}, refresh=True)

            valid_loss = avg_loss_valid / max(valid_count, 1)
            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                if foldername != "":
                    torch.save(model.state_dict(), os.path.join(foldername, "model_best.pth"))
                print(f"[BEST] valid_loss updated: {best_valid_loss:.6f} at epoch {epoch}")

        valid_losses.append(valid_loss)
        elapsed = time.time() - epoch_start_time
        valid_msg = "valid_loss=skip" if np.isnan(valid_loss) else f"valid_loss={valid_loss:.6f}"
        print(f"[Epoch {epoch:03d}/{epochs:03d}] END | train_loss={train_loss:.6f} | {valid_msg} | batches={batch_count} | time={elapsed:.1f}s")

        if foldername != "":
            save_loss_history(foldername, train_losses, valid_losses)
            plot_loss_curve(foldername, train_losses, valid_losses)
            print("[SAVE] loss_history.csv / loss_curve.png updated")

    if foldername != "":
        torch.save(model.state_dict(), output_path)
        print(f"[SAVE] final model: {output_path}")

    total_elapsed = time.time() - total_start_time
    print("\n" + "=" * 90)
    print(f"[TRAIN DONE] total time       : {total_elapsed / 60.0:.2f} min")
    print(f"[TRAIN DONE] final train loss : {train_losses[-1]:.6f}")
    if not np.all(np.isnan(np.asarray(valid_losses, dtype=float))):
        print(f"[TRAIN DONE] best valid loss  : {best_valid_loss:.6f}")
    print("=" * 90 + "\n")

    return train_losses, valid_losses


# -----------------------------
# Plotting
# -----------------------------

def _shade_missing_segments(t, segments, label):
    used = False
    for s, e in segments:
        s = int(s)
        e = int(e)
        if e <= s:
            continue
        plt.axvspan(t[s], t[e - 1], alpha=0.18, label=label if not used else None)
        used = True


def plot_reconstruction(t, true_str, pred_median, pred_min, pred_max, active_start, active_end, xta_peak_idx, missing_segments, save_path, zoom=False):
    apply_plot_style()
    plt.figure(figsize=(13, 5))

    plt.plot(t, true_str, linewidth=1.0, label="True strain")

    valid = ~np.isnan(pred_median)
    if valid.any():
        plt.plot(t[valid], pred_median[valid], linewidth=1.3, label="CSDI median reconstruction")
        plt.fill_between(t[valid], pred_min[valid], pred_max[valid], alpha=0.25, label="Min-max range")

    if active_start is not None and active_end is not None:
        plt.axvspan(t[active_start], t[active_end - 1], alpha=0.08, label="XTa-peak active interval")

    if missing_segments is not None:
        _shade_missing_segments(t, missing_segments, "Removed STr region")

    if xta_peak_idx is not None:
        plt.axvline(t[xta_peak_idx], linestyle="--", linewidth=1.0, label="XTa abs peak")

    if zoom and missing_segments is not None and len(missing_segments) > 0:
        left_idx = min([int(s) for s, _ in missing_segments])
        right_idx = max([int(e) for _, e in missing_segments])
        miss_span = max(1, right_idx - left_idx)
        pad = max(int(2.0 * miss_span), int(0.10 * (active_end - active_start)))
        left = max(0, left_idx - pad)
        right = min(len(t) - 1, right_idx + pad)
        plt.xlim(t[left], t[right])

    plt.xlabel("Time (s)")
    plt.ylabel("Strain (με)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_masked_input(t, true_str, global_missing_mask, active_start, active_end, xta_peak_idx, missing_segments, save_path, zoom=False):
    apply_plot_style()
    observed_visible = true_str.copy().astype(float)
    observed_visible[global_missing_mask] = np.nan

    plt.figure(figsize=(13, 5))
    plt.plot(t, true_str, linewidth=1.0, alpha=0.50, label="True strain")
    plt.plot(t, observed_visible, linewidth=1.2, label="Observed input strain")

    if active_start is not None and active_end is not None:
        plt.axvspan(t[active_start], t[active_end - 1], alpha=0.08, label="XTa-peak active interval")

    if missing_segments is not None:
        _shade_missing_segments(t, missing_segments, "Removed STr region")

    if xta_peak_idx is not None:
        plt.axvline(t[xta_peak_idx], linestyle="--", linewidth=1.0, label="XTa abs peak")

    if zoom and missing_segments is not None and len(missing_segments) > 0:
        left_idx = min([int(s) for s, _ in missing_segments])
        right_idx = max([int(e) for _, e in missing_segments])
        miss_span = max(1, right_idx - left_idx)
        pad = max(int(2.0 * miss_span), int(0.10 * (active_end - active_start)))
        left = max(0, left_idx - pad)
        right = min(len(t) - 1, right_idx + pad)
        plt.xlim(t[left], t[right])

    plt.xlabel("Time (s)")
    plt.ylabel("Strain (με)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _should_plot_scenario(scenario_key, obj, eval_cfg):
    if bool(eval_cfg.get("plot_all_test_scenarios", False)):
        return True
    focus_ratios = eval_cfg.get("focus_plot_ratios", [0.10])
    focus_positions = eval_cfg.get("focus_plot_positions", ["peak_center"])
    focus_modes = eval_cfg.get("focus_plot_modes", ["single_block"])
    tol = 1e-6
    ratio_ok = any(abs(float(obj["missing_ratio"]) - float(r)) < tol for r in focus_ratios)
    pos_ok = obj["position"] in focus_positions
    mode_ok = obj["missing_mode"] in focus_modes
    return ratio_ok and pos_ok and mode_ok


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_and_plot(model, test_loader, test_dataset, nsample=50, foldername="", eval_cfg=None):
    if eval_cfg is None:
        eval_cfg = {}
    os.makedirs(foldername, exist_ok=True)
    plot_dir = os.path.join(foldername, "plots_reconstruction_multiscenario")
    os.makedirs(plot_dir, exist_ok=True)

    device = model.device
    mean = torch.tensor(test_dataset.mean, dtype=torch.float32, device=device)
    std = torch.tensor(test_dataset.std, dtype=torch.float32, device=device)
    str_index = int(test_dataset.str_index)

    raw_events = {e["event_no"]: e for e in test_dataset.events}
    collected = {}
    all_window_outputs = []

    for sample in test_dataset.samples:
        scenario_key = sample["scenario_key"]
        if scenario_key in collected:
            continue
        event = raw_events[int(sample["event_no"])]
        n = event["length"]
        collected[scenario_key] = {
            "event_no": int(sample["event_no"]),
            "missing_mode": sample["missing_mode"],
            "position": sample["position"],
            "missing_ratio": float(sample["missing_ratio"]),
            "values": [[] for _ in range(n)],
            "window_cover_count": np.zeros(n, dtype=int),
            "active_start": int(sample["active_start"]),
            "active_end": int(sample["active_end"]),
            "xta_peak_idx": int(sample["xta_peak_idx"]),
            "missing_start": int(sample["missing_start"]),
            "missing_end": int(sample["missing_end"]),
            "missing_segments": [[int(a), int(b)] for a, b in sample["missing_segments"]],
            "mask_indices": np.asarray(sample["mask_indices"], dtype=np.int64),
        }

    print("\n" + "=" * 90)
    print("[EVAL SETUP] CSDI multi-scenario sampling/evaluation")
    print(f"[EVAL SETUP] nsample            : {nsample}")
    print(f"[EVAL SETUP] test windows       : {len(test_loader.dataset)}")
    print(f"[EVAL SETUP] test scenarios     : {len(collected)}")
    print(f"[EVAL SETUP] diffusion steps    : {model.num_steps}")
    print("=" * 90 + "\n")

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0, dynamic_ncols=True) as it:
            for test_batch in it:
                samples, c_target, eval_points, observed_points, observed_time = model.evaluate(test_batch, nsample)

                samples = samples.permute(0, 1, 3, 2)       # (B, nsample, L, K)
                c_target = c_target.permute(0, 2, 1)        # (B, L, K)
                eval_points = eval_points.permute(0, 2, 1)  # (B, L, K)

                samples_denorm = samples * std.view(1, 1, 1, -1) + mean.view(1, 1, 1, -1)
                target_denorm = c_target * std.view(1, 1, -1) + mean.view(1, 1, -1)

                samples_np = samples_denorm.detach().cpu().numpy()
                target_np = target_denorm.detach().cpu().numpy()
                eval_np = eval_points.detach().cpu().numpy()

                B = samples_np.shape[0]
                for b in range(B):
                    dataset_idx = int(test_batch["scenario_index"][b].item())
                    meta = test_dataset.samples[dataset_idx]
                    scenario_key = meta["scenario_key"]
                    event_no = int(meta["event_no"])
                    start_idx = int(meta["start_idx"])

                    eval_mask = eval_np[b, :, str_index] > 0.5
                    local_indices = np.where(eval_mask)[0]

                    for li in local_indices:
                        gi = start_idx + int(li)
                        vals = samples_np[b, :, li, str_index]
                        collected[scenario_key]["values"][gi].extend(vals.tolist())
                        collected[scenario_key]["window_cover_count"][gi] += 1

                    all_window_outputs.append({
                        "scenario_key": scenario_key,
                        "event_no": event_no,
                        "missing_mode": meta["missing_mode"],
                        "position": meta["position"],
                        "missing_ratio": float(meta["missing_ratio"]),
                        "start_idx": start_idx,
                        "target_str": target_np[b, :, str_index],
                        "eval_mask_str": eval_mask.astype(np.float32),
                        "samples_str": samples_np[b, :, :, str_index],
                    })

    metrics_rows = []
    overall_true = []
    overall_pred = []

    save_reconstruction_csv = bool(eval_cfg.get("save_reconstruction_csv", True))

    for scenario_key, obj in collected.items():
        event_no = int(obj["event_no"])
        event = raw_events[event_no]
        t = event["t"]
        true_str = event["STr"]
        n = len(true_str)

        pred_median = np.full(n, np.nan, dtype=float)
        pred_min = np.full(n, np.nan, dtype=float)
        pred_max = np.full(n, np.nan, dtype=float)
        pred_count = np.zeros(n, dtype=int)

        for i in range(n):
            vals = obj["values"][i]
            if len(vals) > 0:
                arr = np.asarray(vals, dtype=float)
                pred_median[i] = np.median(arr)
                pred_min[i] = np.min(arr)
                pred_max[i] = np.max(arr)
                pred_count[i] = len(arr)

        global_missing_mask = np.zeros(n, dtype=bool)
        mask_indices = np.asarray(obj["mask_indices"], dtype=np.int64)
        mask_indices = mask_indices[(mask_indices >= 0) & (mask_indices < n)]
        global_missing_mask[mask_indices] = True

        valid = (~np.isnan(pred_median)) & global_missing_mask
        rmse = float(np.sqrt(np.mean((pred_median[valid] - true_str[valid]) ** 2))) if valid.any() else np.nan
        mae = float(np.mean(np.abs(pred_median[valid] - true_str[valid]))) if valid.any() else np.nan

        active_start = int(obj["active_start"])
        active_end = int(obj["active_end"])
        active_len = int(active_end - active_start)
        missing_len = int(global_missing_mask.sum())

        if valid.any():
            overall_true.append(true_str[valid])
            overall_pred.append(pred_median[valid])

        metrics_rows.append({
            "scenario_key": scenario_key,
            "event_no": event_no,
            "missing_mode": obj["missing_mode"],
            "position": obj["position"],
            "missing_ratio": float(obj["missing_ratio"]),
            "active_start_idx": active_start,
            "active_end_idx": active_end,
            "active_start_time_s": float(t[active_start]),
            "active_end_time_s": float(t[active_end - 1]),
            "missing_start_idx": int(obj["missing_start"]),
            "missing_end_idx": int(obj["missing_end"]),
            "missing_start_time_s": float(t[int(obj["missing_start"])]),
            "missing_end_time_s": float(t[int(obj["missing_end"]) - 1]),
            "num_reconstructed_points": int(valid.sum()),
            "missing_points": int(missing_len),
            "reconstructed_ratio_of_full": float(valid.sum() / n),
            "reconstructed_ratio_of_active": float(valid.sum() / max(active_len, 1)),
            "rmse": rmse,
            "mae": mae,
        })

        base = safe_name(scenario_key)
        if save_reconstruction_csv:
            csv_path = os.path.join(plot_dir, f"{base}_reconstruction.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["idx", "time_s", "true_strain", "pred_median", "pred_min", "pred_max", "num_values", "is_missing_target"])
                for i in range(n):
                    writer.writerow([i, t[i], true_str[i], pred_median[i], pred_min[i], pred_max[i], pred_count[i], int(global_missing_mask[i])])

        if _should_plot_scenario(scenario_key, obj, eval_cfg):
            plot_reconstruction(
                t, true_str, pred_median, pred_min, pred_max,
                active_start, active_end, int(obj["xta_peak_idx"]), obj["missing_segments"],
                os.path.join(plot_dir, f"{base}_full.png"), zoom=False,
            )
            plot_reconstruction(
                t, true_str, pred_median, pred_min, pred_max,
                active_start, active_end, int(obj["xta_peak_idx"]), obj["missing_segments"],
                os.path.join(plot_dir, f"{base}_missing_zoom.png"), zoom=True,
            )
            plot_masked_input(
                t, true_str, global_missing_mask,
                active_start, active_end, int(obj["xta_peak_idx"]), obj["missing_segments"],
                os.path.join(plot_dir, f"{base}_masked_input_zoom.png"), zoom=True,
            )

    if len(overall_true) > 0:
        y_true = np.concatenate(overall_true)
        y_pred = np.concatenate(overall_pred)
        overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        overall_mae = float(np.mean(np.abs(y_pred - y_true)))
    else:
        overall_rmse = np.nan
        overall_mae = np.nan

    metrics_path = os.path.join(foldername, "reconstruction_metrics_multiscenario.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "scenario_key", "event_no", "missing_mode", "position", "missing_ratio",
            "active_start_idx", "active_end_idx", "active_start_time_s", "active_end_time_s",
            "missing_start_idx", "missing_end_idx", "missing_start_time_s", "missing_end_time_s",
            "num_reconstructed_points", "missing_points", "reconstructed_ratio_of_full",
            "reconstructed_ratio_of_active", "rmse", "mae",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)
        writer.writerow({"scenario_key": "overall", "rmse": overall_rmse, "mae": overall_mae})

    pk_path = os.path.join(foldername, f"generated_outputs_multiscenario_nsample{nsample}.pk")
    with open(pk_path, "wb") as f:
        pickle.dump(all_window_outputs, f)

    print(f"[RESULT] overall RMSE: {overall_rmse:.6f}")
    print(f"[RESULT] overall MAE : {overall_mae:.6f}")
    print(f"[SAVE] metrics: {metrics_path}")
    print(f"[SAVE] plots  : {plot_dir}")
    print("[INFO] default plot setting: only focus scenarios are plotted. All scenarios are still evaluated in metrics CSV.")

    return overall_rmse, overall_mae
