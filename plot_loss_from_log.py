import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("사용법: python plot_loss_from_log.py train_log.txt")
    sys.exit(1)

log_path = sys.argv[1]

epochs = []
train_losses = []
valid_epochs = []
valid_losses = []

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for line in lines:
    m_train = re.search(r"avg_epoch_loss=([0-9eE\.\-]+),\s*epoch=(\d+)", line)
    if m_train:
        train_losses.append(float(m_train.group(1)))
        epochs.append(int(m_train.group(2)))

    m_valid = re.search(r"valid_avg_epoch_loss=([0-9eE\.\-]+),\s*epoch=(\d+)", line)
    if m_valid:
        valid_losses.append(float(m_valid.group(1)))
        valid_epochs.append(int(m_valid.group(2)))

if len(epochs) == 0:
    print("로그에서 train loss를 찾지 못했습니다.")
    sys.exit(1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)

if len(valid_epochs) > 0:
    plt.plot(valid_epochs, valid_losses, label="Valid Loss", linewidth=1.5)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("train_valid_loss.png", dpi=200)
plt.show()

print("저장 완료: train_valid_loss.png")