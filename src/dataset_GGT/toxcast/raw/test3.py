import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import matplotlib.pyplot as plt

# ======================
# Config
# ======================
CSV_PATH = "toxcast_data.csv"   # 改成你的路径
META_COLS = {"smiles", "mol_id", "compound_id"}

# 你要试到几元组合（比如 4 / 5）
MAX_K = 4

# 为了真的能跑：如果某个样本“有值的列”特别多（比如几百），C(m,4) 会非常恐怖
# 这里给个保险阈值：超过就跳过该样本（你可以改大，或设为 None 关闭）
MAX_COLS_PER_ROW = 120

# 画直方图 bins
BINS = 50

# ======================
# Load & preprocess
# ======================
df = pd.read_csv(CSV_PATH)

label_cols = [c for c in df.columns if c not in META_COLS]

# 把 "" 之类的空值处理成 NaN；其余值不管是 0/1/浮点，都算“有值”
Y = df[label_cols].replace("", np.nan)

# 如果你的 CSV 有些列读成字符串，这行会尽量转成数值；转不了的仍保留为 NaN（不影响“是否有值”的判断）
Y = Y.apply(pd.to_numeric, errors="coerce")

# ======================
# 1) Exact pattern counting (key = full set of observed labels)
# ======================
exact_pattern_counter = Counter()

# ======================
# 2) k-itemset counting (key = k-combination of observed labels)
# ======================
k_itemset_counters = {k: Counter() for k in range(2, MAX_K + 1)}

skipped_rows = 0

for idx, row in Y.iterrows():
    observed = [col for col, v in row.items() if not pd.isna(v)]  # “只要有value就算”
    m = len(observed)

    if m == 0:
        continue

    # Exact pattern：不爆炸（key 是整套 observed labels）
    exact_key = tuple(sorted(observed))
    exact_pattern_counter[exact_key] += 1

    # k-itemsets：可能爆炸，所以加个阈值
    if MAX_COLS_PER_ROW is not None and m > MAX_COLS_PER_ROW:
        skipped_rows += 1
        continue

    # 统计 2..K 元组合
    observed_sorted = sorted(observed)
    for k in range(2, MAX_K + 1):
        if m < k:
            break
        for comb in combinations(observed_sorted, k):
            k_itemset_counters[k][comb] += 1

print(f"Total rows: {len(Y)}")
print(f"Skipped rows for k-itemsets due to too many observed labels (> {MAX_COLS_PER_ROW}): {skipped_rows}")

# ======================
# Helper: plot histogram of counts
# ======================
def plot_hist(counter: Counter, title: str):
    if len(counter) == 0:
        print(f"[WARN] empty counter: {title}")
        return
    counts = np.array(list(counter.values()))
    plt.figure(figsize=(7,4))
    plt.hist(counts, bins=BINS)
    plt.xlabel("Occurrence count")
    plt.ylabel("Number of patterns")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# ======================
# Plot histograms
# ======================
plot_hist(exact_pattern_counter, "Exact co-measured label-set patterns (full set per compound)")

for k in range(2, MAX_K + 1):
    plot_hist(k_itemset_counters[k], f"{k}-itemset co-measured patterns")

# 可选：y 轴取 log，更容易看 heavy-tail
def plot_hist_logy(counter: Counter, title: str):
    if len(counter) == 0:
        return
    counts = np.array(list(counter.values()))
    plt.figure(figsize=(7,4))
    plt.hist(counts, bins=BINS)
    plt.yscale("log")
    plt.xlabel("Occurrence count")
    plt.ylabel("Number of patterns (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# 你如果想要 log 版，把下面取消注释
# plot_hist_logy(exact_pattern_counter, "Exact patterns (log y)")
# for k in range(2, MAX_K + 1):
#     plot_hist_logy(k_itemset_counters[k], f"{k}-itemsets (log y)")

# ======================
# Export top patterns (看最常见的“共同出现集合”)
# ======================
def top_n(counter: Counter, n=20):
    return counter.most_common(n)

print("\nTop exact patterns:")
for key, cnt in top_n(exact_pattern_counter, 10):
    print(cnt, " | ", list(key)[:10], ("... (truncated)" if len(key) > 10 else ""))

for k in range(2, MAX_K + 1):
    print(f"\nTop {k}-itemsets:")
    for key, cnt in top_n(k_itemset_counters[k], 10):
        print(cnt, " | ", key)

# 可选：保存成 CSV（2-itemset 最常用）
k2 = k_itemset_counters[2]
k2_df = pd.DataFrame([(a, b, c) for (a, b), c in k2.items()], columns=["label_1", "label_2", "count"])
k2_df.sort_values("count", ascending=False).to_csv("toxcast_co_measured_pairs.csv", index=False)
print("\nSaved: toxcast_co_measured_pairs.csv")
