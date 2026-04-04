from scipy.io import loadmat
import numpy as np

# 1) 파일 불러오기
data = loadmat("resp_total_re_05.mat")
records = data["resp_total_re_05"]

print("=" * 50)
print("Loaded file: resp_total_re_05.mat")
print("records shape:", records.shape)
print("=" * 50)

# 2) 각 record의 shape 확인
for i in range(20):
    rec = records[0, i]

    acc_g = np.asarray(rec["acc_g"]).squeeze()
    XTa   = np.asarray(rec["XTa"]).squeeze()
    XT    = np.asarray(rec["XT"]).squeeze()
    strr  = np.asarray(rec["str"]).squeeze()

    print(f"\n[Record {i+1}]")
    print("acc_g shape:", acc_g.shape)
    print("XTa   shape:", XTa.shape)
    print("XT    shape:", XT.shape)
    print("str   shape:", strr.shape)

# 3) 1번 record를 예시로 X, Y 만들기
print("\n" + "=" * 50)
print("Example: make X and Y from record 1")
print("=" * 50)

rec = records[0, 0]

acc_g = np.asarray(rec["acc_g"]).squeeze()
XTa   = np.asarray(rec["XTa"]).squeeze()
XT    = np.asarray(rec["XT"]).squeeze()
strr  = np.asarray(rec["str"]).squeeze()

X = np.column_stack([acc_g, XTa, XT])   # (T, 3)
Y = strr                                # (T,)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

print("\nFirst 5 rows of X:")
print(X[:5])

print("\nFirst 5 values of Y:")
print(Y[:5])