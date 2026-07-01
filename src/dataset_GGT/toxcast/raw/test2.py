import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
CSV_PATH = "toxcast_data.csv"     # <- 改成你的文件
META_COLS = {"smiles", "mol_id", "compound_id"}

TOPK = 80                    # 画图任务数（ToxCast 建议 60~120）
MIN_OVERLAP = 50             # 计算相关性的最小共标注样本数
PHI_THRESH = 0.6             # 高相关阈值（画框 + top pairs）

# -----------------------
# Helpers
# -----------------------
def pairwise_phi(Y, min_overlap=50):
    """
    Y: [N, T] float with NaN for missing; values should be 0/1 for labeled rows
    Returns:
      corr: [T,T] phi coefficient matrix (NaN if insufficient overlap or degenerate)
      overlap: [T,T] co-annotation counts
    """
    N, T = Y.shape
    corr = np.full((T, T), np.nan, dtype=float)
    overlap = np.zeros((T, T), dtype=int)

    for i in range(T):
        yi = Y[:, i]
        for j in range(i, T):
            yj = Y[:, j]
            m = ~np.isnan(yi) & ~np.isnan(yj)
            n = int(m.sum())
            overlap[i, j] = overlap[j, i] = n
            if n < min_overlap:
                continue

            a = yi[m].astype(int)
            b = yj[m].astype(int)

            n11 = np.sum((a == 1) & (b == 1))
            n10 = np.sum((a == 1) & (b == 0))
            n01 = np.sum((a == 0) & (b == 1))
            n00 = np.sum((a == 0) & (b == 0))

            den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
            if den == 0:
                continue
            corr_ij = (n11 * n00 - n10 * n01) / den
            corr[i, j] = corr[j, i] = corr_ij

    np.fill_diagonal(corr, 1.0)
    return corr, overlap

def top_pairs_table(corr, overlap, labels, phi_thresh=0.6, topn=50):
    rows = []
    T = corr.shape[0]
    for i in range(T):
        for j in range(i + 1, T):
            v = corr[i, j]
            if np.isnan(v):
                continue
            if abs(v) >= phi_thresh:
                rows.append((labels[i], labels[j], float(v), int(overlap[i, j])))
    rows.sort(key=lambda x: abs(x[2]), reverse=True)
    return pd.DataFrame(rows, columns=["task_1", "task_2", "phi", "n_overlap"]).head(topn)

def plot_heatmap(mat, title, vmin=None, vmax=None, cmap="coolwarm"):
    plt.figure(figsize=(11, 9))
    im = plt.imshow(mat, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def plot_corr_with_boxes(corr, overlap, title, phi_thresh=0.6, min_overlap=50):
    """
    correlation heatmap + draw black boxes for |phi|>=phi_thresh and overlap>=min_overlap
    """
    plt.figure(figsize=(11, 9))
    im = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(im, fraction=0.046, pad=0.04)

    T = corr.shape[0]
    # draw boxes
    for i in range(T):
        for j in range(T):
            v = corr[i, j]
            if np.isnan(v):
                continue
            if overlap[i, j] >= min_overlap and abs(v) >= phi_thresh and i != j:
                # rectangle outline at (j,i)
                plt.gca().add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2)
                )

    plt.tight_layout()
    plt.show()

# -----------------------
# Main
# -----------------------
df = pd.read_csv(CSV_PATH)

label_cols = [c for c in df.columns if c not in META_COLS]

# 1) convert to numeric label matrix (missing -> NaN)
Y_df = df[label_cols].replace("", np.nan)
Y = Y_df.astype(float).to_numpy()

# 2) choose Top-K tasks by coverage (non-missing count)
coverage = np.sum(~np.isnan(Y), axis=0)
topk_idx = np.argsort(coverage)[-TOPK:]
topk_idx = np.sort(topk_idx)  # keep stable-ish order (optional)

labels_k = [label_cols[i] for i in topk_idx]
Yk = Y[:, topk_idx]

# 3) compute correlation + overlap
corr_k, overlap_k = pairwise_phi(Yk, min_overlap=MIN_OVERLAP)

# 4) save matrices
pd.DataFrame(corr_k, index=labels_k, columns=labels_k).to_csv("toxcast_phi_corr_topk.csv")
pd.DataFrame(overlap_k, index=labels_k, columns=labels_k).to_csv("toxcast_overlap_topk.csv")

# 5) export top correlated pairs (paper-friendly)
top_pairs = top_pairs_table(corr_k, overlap_k, labels_k, phi_thresh=PHI_THRESH, topn=100)
top_pairs.to_csv("toxcast_top_correlated_pairs.csv", index=False)

print("Saved:")
print("  toxcast_phi_corr_topk.csv")
print("  toxcast_overlap_topk.csv")
print("  toxcast_top_correlated_pairs.csv")
print("\nTop correlated pairs preview:")
print(top_pairs.head(20).to_string(index=False))

# 6) plots
plot_corr_with_boxes(
    corr_k, overlap_k,
    title=f"ToxCast Phi Correlation (Top-{TOPK} by coverage) | boxes: |phi|≥{PHI_THRESH}, overlap≥{MIN_OVERLAP}",
    phi_thresh=PHI_THRESH,
    min_overlap=MIN_OVERLAP
)

plot_heatmap(
    overlap_k,
    title=f"ToxCast Co-annotation Counts (Top-{TOPK} by coverage)",
    vmin=0, vmax=np.nanmax(overlap_k),
    cmap="viridis"
)
