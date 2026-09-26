import os
import re
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

# ===== 配置 =====
DATA_PATH = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrc_class5_63468.txt"  # 每行: 序列\t等级(0-4)
OUT_ROOT  = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrc_processed_5fold"
SEQ_LEN   = 80          # 你的新数据是 ~80bp，这里设为 80
NUM_CLASSES = 5
VAL_RATIO = 0.2
KFOLD     = 5
SEED      = 42

random.seed(SEED)
np.random.seed(SEED)

# ===== 读取(序列 + 等级 0-4) =====
seq_str_list, label_list = [], []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for ln, line in enumerate(tqdm(f, desc="Reading"), start=1):
        s = line.strip()
        if not s:
            continue
        parts = re.split(r"[\t ]+", s)
        if len(parts) < 2:
            continue
        seq, lab_str = parts[0], parts[1]

        # 处理表头，如 "Sequence\tClass"
        if (not lab_str.replace(".", "", 1).isdigit()) and lab_str.lower() in {"class", "label"}:
            continue

        try:
            lab = int(float(lab_str))
        except ValueError:
            m = re.search(r"(\d+)", lab_str)
            if not m:
                raise
            lab = int(m.group(1))

        if not (0 <= lab < NUM_CLASSES):
            raise ValueError(f"Label out of range at line {ln}: {lab}")

        seq = seq.upper().replace("U", "T")  # 有些数据可能有 U，统一成 T
        seq_str_list.append(seq)
        label_list.append(lab)

y_all = np.asarray(label_list, dtype=np.int64)

print("总样本数:", len(seq_str_list))
uniq, cnt = np.unique(y_all, return_counts=True)
print("Label 分布:", dict(zip(uniq.tolist(), cnt.tolist())))

# ===== one-hot 到 (N, SEQ_LEN, DEPTH=5) =====
PAD_TOKEN = "N"
vocab = ['A', 'C', 'G', 'T']
NUM_OOV = 1          # N/OOV通道
tok2idx = {ch: i + NUM_OOV for i, ch in enumerate(vocab)}
DEPTH = len(vocab) + NUM_OOV  # =5


def pad_or_truncate(chars, L=SEQ_LEN, pad_value=PAD_TOKEN):
    """把字符序列 pad/truncate 到固定长度 L"""
    return chars[:L] if len(chars) >= L else chars + [pad_value] * (L - len(chars))


def one_hot_from_str(seq_str):
    """把原始字符串序列 -> (SEQ_LEN, DEPTH)"""
    chars = list(seq_str)
    chars = pad_or_truncate(chars, L=SEQ_LEN)
    arr = np.zeros((SEQ_LEN, DEPTH), dtype=np.float32)
    for i, ch in enumerate(chars):
        idx = tok2idx.get(ch, 0)  # 其它/N -> 0
        arr[i, idx] = 1.0
    return arr


X_all = np.stack([one_hot_from_str(s) for s in seq_str_list], axis=0).astype(np.float32)
print("Encoded X:", X_all.shape, " y:", y_all.shape)

# ===== 计算 CGR 特征 (作为 extra 特征) =====
# 使用二维 Chaos Game Representation：
#   A: (0,0), C:(0,1), G:(1,1), T:(1,0)
#   从 (0.5,0.5) 开始迭代，每次取中点。
#   然后将所有点映射到 n_bins x n_bins 网格中，计数并归一化。

CGR_BINS = 16                     # 16x16 网格
CGR_DIM = CGR_BINS * CGR_BINS
base2coord = {
    'A': (0.0, 0.0),
    'C': (0.0, 1.0),
    'G': (1.0, 1.0),
    'T': (1.0, 0.0)
}


def cgr_from_str(seq_str, n_bins=CGR_BINS):
    # 去掉非 A/C/G/T 的字符（N 等）
    seq = [ch for ch in seq_str if ch in base2coord]
    if len(seq) == 0:
        # 如果全是 N，就返回全 0
        return np.zeros((n_bins * n_bins,), dtype=np.float32)

    x, y = 0.5, 0.5
    grid = np.zeros((n_bins, n_bins), dtype=np.float32)

    for ch in seq:
        vx, vy = base2coord[ch]
        x = (x + vx) / 2.0
        y = (y + vy) / 2.0
        ix = int(x * n_bins)
        iy = int(y * n_bins)
        if ix == n_bins:
            ix = n_bins - 1
        if iy == n_bins:
            iy = n_bins - 1
        grid[iy, ix] += 1.0

    total = grid.sum()
    if total > 0:
        grid /= total  # 归一化为频率
    return grid.reshape(-1).astype(np.float32)


print("Computing CGR features...")
cgr_all = np.stack([cgr_from_str(s) for s in tqdm(seq_str_list, desc="CGR")], axis=0)
print("CGR_all shape:", cgr_all.shape)  # (N, CGR_DIM)

os.makedirs(OUT_ROOT, exist_ok=True)


def save_pair(tr_idx, va_idx, out_dir, dump_lists=False):
    """
    同时保存：
    - train_dataset.pt / val_dataset.pt （用于训练，包含 X / extra / y）
    - train.csv / val.csv （便于查看）
    """
    os.makedirs(out_dir, exist_ok=True)

    import torch

    # 1) 切分 X / extra / y
    Xtr = X_all[tr_idx].astype(np.float32)
    Xva = X_all[va_idx].astype(np.float32)
    Etr = cgr_all[tr_idx].astype(np.float32)   # extra 特征 (CGR)
    Eva = cgr_all[va_idx].astype(np.float32)
    ytr = y_all[tr_idx].astype(np.int64)
    yva = y_all[va_idx].astype(np.int64)

    # 2) 保存 .pt （训练真正用的）
    torch.save(
        {"X": Xtr, "extra": Etr, "y": ytr},
        os.path.join(out_dir, "train_dataset.pt")
    )
    torch.save(
        {"X": Xva, "extra": Eva, "y": yva},
        os.path.join(out_dir, "val_dataset.pt")
    )

    # 3) 保存 CSV（原始序列 + 标签，便于肉眼看）
    train_seqs = [seq_str_list[i] for i in tr_idx]
    val_seqs   = [seq_str_list[i] for i in va_idx]

    df_tr = pd.DataFrame({
        "seq": train_seqs,
        "label": y_all[tr_idx]
    })
    df_va = pd.DataFrame({
        "seq": val_seqs,
        "label": y_all[va_idx]
    })

    df_tr.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    df_va.to_csv(os.path.join(out_dir, "val.csv"), index=False)

    # 4) 可选：保存整个列表（有时方便 debug）
    if dump_lists:
        pickle.dump(seq_str_list, open(os.path.join(out_dir, "all_seq_list.pkl"), "wb"))
        pickle.dump(label_list,  open(os.path.join(out_dir, "all_label_list.pkl"), "wb"))

    print(f"[SAVE] .pt + .csv -> {out_dir}")


# ===== 1) 分层 8:2 (可作为最终测试集) =====
sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
tr_idx, va_idx = next(sss.split(X_all, y_all))

holdout_dir = os.path.join(OUT_ROOT, f"preprocessed_data_class{NUM_CLASSES}_63468_8_2")
save_pair(tr_idx, va_idx, holdout_dir, dump_lists=True)

# ===== 2) 5 折 k-fold 分层 =====
kroot = os.path.join(OUT_ROOT, f"kfold{KFOLD}_class{NUM_CLASSES}")
os.makedirs(kroot, exist_ok=True)

skf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
for i, (tr, va) in enumerate(skf.split(X_all, y_all), start=1):
    fold_dir = os.path.join(kroot, f"fold{i}")
    save_pair(tr, va, fold_dir, dump_lists=False)
    print(f"[KFOLD] fold{i} -> {fold_dir}  (Ntr={len(tr)}, Nva={len(va)})")

print("[DONE] preprocess OK.")

