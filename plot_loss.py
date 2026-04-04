import re
import os
import matplotlib.pyplot as plt

LOG_FILE = "train_log.txt"
SAVE_DIR = "plot_results"
SAVE_PATH = os.path.join(SAVE_DIR, "loss_curve.png")

os.makedirs(SAVE_DIR, exist_ok=True)

if not os.path.isfile(LOG_FILE):
    raise FileNotFoundError(
        "train_log.txt 파일이 없습니다. nonlinear 폴더 안에 train_log.txt를 넣어주세요."
    )

with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# train loss 추출
train_pattern = r"avg_epoch_loss=([0-9.]+), epoch=([0-9]+)"
train_matches = re.findall(train_pattern, text)

# valid loss 추출
valid_pattern = r"valid_avg_epoch_loss=([0-9.]+), epoch=([0-9]+)"
valid_matches = re.findall(valid_pattern, text)

if len(train_matches) == 0:
    raise ValueError("train_log.txt에서 train loss를 찾지 못했습니다.")

train_dict = {}
for loss, epoch in train_matches:
    train_dict[int(epoch)] = float(loss)

valid_dict = {}
for loss, epoch in valid_matches:
    valid_dict[int(epoch)] = float(loss)

train_epochs = sorted(train_dict.keys())
train_losses = [train_dict[e] for e in train_epochs]

valid_epochs = sorted(valid_dict.keys())
valid_losses = [valid_dict[e] for e in valid_epochs]

print("Train points:", len(train_epochs))
print("Valid points:", len(valid_epochs))

plt.figure(figsize=(10, 6))
plt.plot(train_epochs, train_losses, linewidth=2, label="Train Loss")
if len(valid_epochs) > 0:
    plt.plot(valid_epochs, valid_losses, marker="o", linewidth=2, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(SAVE_PATH, dpi=200)
plt.show()

print(f"Saved: {SAVE_PATH}")