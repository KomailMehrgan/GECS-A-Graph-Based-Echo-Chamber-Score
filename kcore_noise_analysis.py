"""
K-core noise analysis for GECS pilot communities.

For each pilot community, the retweet graph is progressively filtered with
k-core decomposition (k = 1..10): nodes with weighted degree < k are
iteratively removed, and the political-stance distribution is recomputed
on the surviving ("classified") users at every k.

This script consumes the *already computed* per-k stance distributions
(stance_each_community.json) -- one entry per community, each holding the
label distribution over ten k-core thresholds -- and:

  1. Recomputes the Dominant Stance Share (DSS) at every k, using the same
     definition as the rest of the paper: DSS = max(black%, white%) /
     (black% + white%), i.e. the share of the dominant stance among users
     who received one of the two clear political-stance labels. Users with
     no label ("") and users in the grey/leaning-grey zone are excluded
     from the denominator, exactly as in Section IV-E.
  2. Builds a tidy table (community x k -> total_users, DSS, unclassified%).
  3. Plots DSS as a function of k for all ten communities, to be used as
     the new figure after Fig. 6.
  4. Prints a focused before/after summary for community 17381, the case
     discussed in the text.

Usage:
    python kcore_noise_analysis.py stance_each_community.json --outdir out/
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

BLACK_LABEL = "سیاه"   # "black" stance
WHITE_LABEL = "سفید"   # "white" (opposing) stance
UNCLASSIFIED_LABEL = ""  # no stance label recovered for the user


def load_records(json_path: str) -> list[dict]:
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    return [entry["json_result"] for entry in raw]


def dss_from_distribution(distribution: dict) -> float | None:
    """DSS = max(black, white) / (black + white), restricted to the two
    clear stance labels. Returns None if neither label is present at all
    (can't compute a meaningful DSS)."""
    black = distribution.get(BLACK_LABEL, 0.0)
    white = distribution.get(WHITE_LABEL, 0.0)
    denom = black + white
    if denom == 0:
        return None
    return 100.0 * max(black, white) / denom


def build_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        cid = rec["community_id"]
        for th in rec["thresholds"]:
            k = th["k"]
            dist = th["distribution"]
            rows.append(
                {
                    "community_id": cid,
                    "k": k,
                    "total_users": th["total_users"],
                    "unclassified_pct": dist.get(UNCLASSIFIED_LABEL, 0.0),
                    "black_pct": dist.get(BLACK_LABEL, 0.0),
                    "white_pct": dist.get(WHITE_LABEL, 0.0),
                    "dss": dss_from_distribution(dist),
                }
            )
    return pd.DataFrame(rows)


def plot_dss_vs_k(df: pd.DataFrame, out_path: Path, highlight: int = 17381) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cid, grp in df.groupby("community_id"):
        grp = grp.sort_values("k")
        is_highlight = cid == highlight
        ax.plot(
            grp["k"],
            grp["dss"],
            marker="o",
            markersize=3,
            linewidth=2.4 if is_highlight else 1.0,
            color="#d62728" if is_highlight else "#9aa5b1",
            alpha=1.0 if is_highlight else 0.7,
            zorder=3 if is_highlight else 2,
            label=f"Community {cid}" if is_highlight else None,
        )
    ax.set_xlabel("k-core threshold (k)")
    ax.set_ylabel("Dominant Stance Share, DSS (%)")
    ax.set_title("DSS vs. k-core noise-filtering threshold, all pilot communities")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def print_case_study(df: pd.DataFrame, community_id: int) -> None:
    sub = df[df["community_id"] == community_id].sort_values("k")
    print(f"\nCase study: community {community_id}")
    print(
        f"{'k':>3} {'users':>8} {'unclassified%':>14} "
        f"{'black%':>8} {'white%':>8} {'DSS%':>7}"
    )
    for _, r in sub.iterrows():
        dss = f"{r['dss']:.1f}" if r["dss"] is not None else "n/a"
        print(
            f"{r['k']:>3} {r['total_users']:>8} {r['unclassified_pct']:>14.2f} "
            f"{r['black_pct']:>8.2f} {r['white_pct']:>8.2f} {dss:>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--highlight", type=int, default=17381)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = load_records(args.json_path)
    df = build_table(records)
    df.to_csv(outdir / "kcore_dss_table.csv", index=False)

    plot_dss_vs_k(df, outdir / "fig_kcore_noise_analysis.png", highlight=args.highlight)
    print_case_study(df, args.highlight)

    # Compact summary at k=1 vs. best-DSS k (excluding k where the surviving
    # sample is too small to be meaningful, i.e. total_users < 30).
    print("\nSummary: DSS at k=1 vs. peak DSS at a k with >=30 surviving users")
    summary_rows = []
    for cid, grp in df.groupby("community_id"):
        grp = grp.sort_values("k")
        k1 = grp[grp["k"] == 1].iloc[0]
        reliable = grp[grp["total_users"] >= 30]
        if reliable.empty:
            continue
        peak = reliable.loc[reliable["dss"].idxmax()]
        summary_rows.append(
            {
                "community_id": cid,
                "dss_k1": round(k1["dss"], 1) if k1["dss"] is not None else None,
                "peak_k": int(peak["k"]),
                "dss_peak": round(peak["dss"], 1) if peak["dss"] is not None else None,
                "n_at_peak": int(peak["total_users"]),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("community_id")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(outdir / "kcore_dss_summary.csv", index=False)


if __name__ == "__main__":
    main()
