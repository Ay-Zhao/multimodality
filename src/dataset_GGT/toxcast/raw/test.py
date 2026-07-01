import re
import pandas as pd
from collections import Counter

CSV_PATH = "toxcast_data.csv"   # <- 改这里

META_COLS = {"smiles", "mol_id", "compound_id"}

# 一些常见时间写法：80hr / 24h / 1440 / 0120 / 48hr / 120hpf / 144hpf ...
TIME_PATTERNS = [
    r"^\d+(?:hr|h|H)$",        # 80hr, 24h, 48hr
    r"^\d+hpf$",               # 120hpf, 144hpf
    r"^\d{3,4}$",              # 0120, 0480, 0960, 1440 (分钟类常见)
]

DIRECTION_TOKENS = {
    "up", "dn", "down", "pos", "positive", "neg", "negative",
    "agonist", "antagonist", "inhib", "inhibition", "activation"
}

def looks_like_time(tok: str) -> bool:
    t = tok.lower()
    return any(re.match(p, t) for p in TIME_PATTERNS)

def normalize_direction(tok: str):
    t = tok.lower()
    mapping = {
        "dn": "down",
        "down": "down",
        "up": "up",
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
    }
    if t in mapping:
        return mapping[t]
    if t in {"agonist", "antagonist", "inhib", "inhibition", "activation"}:
        return t
    return None

def parse_label(col: str) -> dict:
    """
    Parse a ToxCast-style column name into structured parts.

    Output fields:
      - raw: original column
      - prefix: platform prefix (ACEA/APR/ATG/BSK/NVS/OT/TOX21/Tanguay/...)
      - tokens: all tokens split by "_"
      - cell_or_system: best guess of cell line/system token (if present)
      - time: best guess time token (if present)
      - direction: up/down/positive/negative/agonist/antagonist/... (if present)
      - mode: CIS/TRANS (ATG common), else None
      - endpoint: remaining tokens joined (best-effort “what is being measured”)
    """
    tokens = col.split("_")
    prefix = tokens[0] if tokens else None

    # Identify CIS/TRANS if present (common in ATG)
    mode = None
    for t in tokens:
        if t.upper() in {"CIS", "TRANS"}:
            mode = t.upper()
            break

    # Identify time token: first token that matches time patterns
    time_tok = None
    for t in tokens:
        if looks_like_time(t):
            time_tok = t
            break

    # Identify direction token: usually last token, but not always (e.g., Tox21 has ..._viability)
    direction = None
    # check from end to start for a direction-ish token
    for t in reversed(tokens):
        d = normalize_direction(t)
        if d is not None:
            direction = d
            break

    # Best-guess "cell or system" token:
    # Heuristic: second token often is cell line/system for many prefixes (APR_HepG2..., ACEA_T47D..., CEETOX_H295R..., NCCT_HEK293T...)
    cell_or_system = None
    if len(tokens) >= 2:
        # For BSK, tokens[1] like 3C/4H/BE3C... are system codes; still useful as "system"
        cell_or_system = tokens[1]

    # Build endpoint = tokens excluding prefix, cell/system, time, direction, mode
    drop = set()
    drop.add(prefix)
    if cell_or_system: drop.add(cell_or_system)
    if time_tok: drop.add(time_tok)
    if mode: drop.add(mode)
    # drop the exact direction token string if it exists in tokens (case-insensitive match)
    if direction:
        for t in tokens:
            if normalize_direction(t) == direction:
                drop.add(t)

    endpoint_tokens = [t for t in tokens if t not in drop]
    endpoint = "_".join(endpoint_tokens) if endpoint_tokens else ""

    return {
        "raw": col,
        "prefix": prefix,
        "tokens": tokens,
        "cell_or_system": cell_or_system,
        "time": time_tok,
        "direction": direction,
        "mode": mode,
        "endpoint": endpoint,
    }

def main():
    df = pd.read_csv(CSV_PATH)
    label_cols = [c for c in df.columns if c not in META_COLS]

    # Parse all labels
    parsed = pd.DataFrame([parse_label(c) for c in label_cols])

    # Extra: platform counts
    prefix_counts = parsed["prefix"].value_counts().rename_axis("prefix").reset_index(name="n_tasks")

    # Extra: missingness / coverage per task (how many non-missing labels)
    # (useful later for correlation analysis)
    coverage = df[label_cols].notna().sum(axis=0).rename("n_labeled").reset_index().rename(columns={"index": "raw"})
    parsed2 = parsed.merge(coverage, on="raw", how="left")

    # Candidate “redundancy pairs” within same (prefix, cell/system, endpoint) but different time/direction
    # This is great for spotting clusters like APR_HepG2_*_24h_up vs *_72h_up etc.
    groups = parsed2.groupby(["prefix", "cell_or_system", "endpoint"], dropna=False)
    redundant_rows = []
    for (p, cs, ep), g in groups:
        if len(g) >= 2:
            # only keep groups with variation in time/direction
            if g["time"].nunique(dropna=False) > 1 or g["direction"].nunique(dropna=False) > 1:
                redundant_rows.append(g.sort_values(["time", "direction", "raw"]))
    redundant = pd.concat(redundant_rows, ignore_index=True) if redundant_rows else parsed2.head(0)

    # Save outputs
    parsed2.sort_values(["prefix", "cell_or_system", "endpoint", "time", "direction", "raw"], inplace=True)

    parsed2.to_csv("toxcast_labels_parsed.csv", index=False)
    prefix_counts.to_csv("toxcast_prefix_counts.csv", index=False)
    redundant.to_csv("toxcast_redundant_candidates.csv", index=False)

    print("Saved:")
    print("  toxcast_labels_parsed.csv")
    print("  toxcast_prefix_counts.csv")
    print("  toxcast_redundant_candidates.csv")
    print()
    print("Top prefixes:")
    print(prefix_counts.head(15).to_string(index=False))
    print()
    print("Example parsed rows:")
    print(parsed2.head(10)[["raw","prefix","cell_or_system","mode","time","direction","endpoint","n_labeled"]].to_string(index=False))

if __name__ == "__main__":
    main()
