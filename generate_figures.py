

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. SETTINGS
# =====================================================================

STANCE_JSON = "data/stance_each_community.json"   # per-user political-stance data
GENERAL_JSON = "data/Generall_info.json"          # overall corpus statistics
OUTPUT_DIR = Path("output_figures")


CHAMBER_IDS = [22493, 31969, 17287, 21579, 9053, 17192, 1000, 27903, 17526]

if len(sys.argv) > 1:
    CHAMBER_IDS = [int(x) for x in sys.argv[1:]]

K_THRESHOLD = 1          # evidence threshold k used for the stance figures (Figs 8-9)
STABLE_EXAMPLE = 9053    # "large, stable" example chamber for Fig. 7
UNSTABLE_EXAMPLE = 17381 # "shallow evidence" example chamber for Fig. 7

CECS_JSON = "data/final_cecs_scores.json"         # per-chamber CECS scores (Figs 2-6)

# This gets filled in by load_cecs_data() in main() - see below.
CECS_DATA = {}


STANCE_A = "سیاه"
STANCE_B = "سفید"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# =====================================================================
# 2. LOAD DATA
# =====================================================================

def load_general_info(path):
    """Read Generall_info.json and print it (this is Table II in the paper,
    a single summary table, not a per-chamber figure)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)[0]
    print("\n=== Overall corpus statistics (Generall_info.json) ===")
    for key, value in data.items():
        print(f"  {key}: {value}")
    print()


def load_stance_data(path):
    """Read stance_each_community.json and return {community_id: json_result}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {item["json_result"]["community_id"]: item["json_result"] for item in raw}


def load_cecs_data(path):
    """Read final_cecs_scores.json and return {chamber_id: {NI, ND, NDeaf, NFix, CECS}}."""
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    data = {}
    for row in rows:
        data[row["Chamber_ID"]] = {
            "NI": row["Isolation_Score"],
            "ND": row["Density_Score"],
            "NDeaf": row["Deafness_Score"],
            "NFix": row["Fixation_Score"],
            "CECS": row["CECS"],
            "Mod": row.get("Modularity_Score"),
        }

    # Density_Score comes in as a raw clustering coefficient (roughly 0-1),
    # while the other three dimensions are already on a 0-100 scale. If we
    # plot it as-is it will look like a flat line at zero next to the others.
    # Detect that and rescale it x100 *for plotting only*, and say so.
    max_density = max(v["ND"] for v in data.values())
    if max_density <= 1.5:
        print(f"NOTE: Density_Score looks like it's on a 0-1 scale (max={max_density:.3f}), "
              f"not 0-100 like the other dimensions. Multiplying by 100 for the figures.")
        for v in data.values():
            v["ND"] *= 100

    return data


def get_stance_row(stance_by_id, cid, k):
    """Return the stance breakdown for one chamber at threshold k."""
    thresholds = stance_by_id[cid]["thresholds"]
    row = next(t for t in thresholds if t["k"] == k)
    dist = row["distribution"]
    a = dist.get(STANCE_A, 0)
    b = dist.get(STANCE_B, 0)
    unclassified = dist.get("", 0)
    gray = 100 - a - b - unclassified   # everything in between the two poles
    dss = max(a, b) / (a + b) * 100 if (a + b) > 0 else None
    return {"cid": cid, "total_users": row["total_users"],
            "a": a, "b": b, "gray": gray, "unclassified": unclassified, "dss": dss}


def filter_available(chamber_ids, data_dict, label):
    """Keep only chamber IDs that actually have data, and warn about the rest."""
    available = [c for c in chamber_ids if c in data_dict]
    missing = [c for c in chamber_ids if c not in data_dict]
    if missing:
        print(f"NOTE: {missing} not found in {label} data - skipping them for that figure.")
    return available


# =====================================================================
# 3. STRUCTURAL FIGURES (Figs. 2-6, use CECS_DATA)
# =====================================================================

def fig2_dimensions_bar(chamber_ids):
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    ids = sorted(ids, key=lambda c: -CECS_DATA[c]["CECS"])
    x = np.arange(len(ids))
    width = 0.2
    dims = ["NI", "ND", "NDeaf", "NFix"]
    names = ["Isolation", "Internal Density", "Deafness", "Fixation"]
    colors = ["#1f3a5f", "#2a9d8f", "#c1272d", "#e0a324"]

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=200)
    for i, (dim, name, color) in enumerate(zip(dims, names, colors)):
        values = [CECS_DATA[c][dim] for c in ids]
        ax.bar(x + (i - 1.5) * width, values, width=width, color=color, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in ids], rotation=30)
    ax.set_xlabel("Chamber ID (ranked by CECS, descending)")
    ax.set_ylabel("Normalized score (0-100)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_dimensions.png", bbox_inches="tight")
    plt.close(fig)


def fig3_cecs_ranking(chamber_ids):
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    ids = sorted(ids, key=lambda c: -CECS_DATA[c]["CECS"])
    scores = [CECS_DATA[c]["CECS"] for c in ids]

    fig, ax = plt.subplots(figsize=(6.3, 3.6), dpi=200)
    cmap = plt.cm.RdYlBu_r
    norm = (np.array(scores) - min(scores)) / (max(scores) - min(scores) + 1e-9)
    bars = ax.bar([str(c) for c in ids], scores, color=cmap(norm), edgecolor="#333333")
    for b, v in zip(bars, scores):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Chamber ID")
    ax.set_ylabel("CECS (0-100)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_ranking.png", bbox_inches="tight")
    plt.close(fig)


def fig4_correlation_heatmap(chamber_ids):
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    if len(ids) < 2:
        print("NOTE: need at least 2 chambers with CECS data for Fig. 4 - skipping.")
        return
    dims = {"Isolation": "NI", "Density": "ND", "Deafness": "NDeaf", "Fixation": "NFix"}
    names = list(dims.keys())
    matrix = np.zeros((4, 4))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            va = [CECS_DATA[c][dims[a]] for c in ids]
            vb = [CECS_DATA[c][dims[b]] for c in ids]
            matrix[i, j] = np.corrcoef(va, vb)[0, 1]

    fig, ax = plt.subplots(figsize=(4.6, 4.1), dpi=200)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(names)
    for i in range(4):
        for j in range(4):
            color = "white" if abs(matrix[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_correlation.png", bbox_inches="tight")
    plt.close(fig)


def fig5_radar_high_vs_low(chamber_ids):
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    if len(ids) < 2:
        print("NOTE: need at least 2 chambers with CECS data for Fig. 5 - skipping.")
        return
    ranked = sorted(ids, key=lambda c: -CECS_DATA[c]["CECS"])
    n = min(3, len(ranked) // 2) or 1          # up to 3 highest vs 3 lowest
    top = ranked[:n]
    bottom = ranked[-n:]

    labels = ["Isolation", "Density", "Deafness", "Fixation"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.2, 5.2), dpi=200)
    ax = fig.add_subplot(111, polar=True)
    high_colors = ["#1f3a5f", "#c1272d", "#e0a324"]
    low_colors = ["#2a9d8f", "#888888", "#8b5a2b"]

    def plot_one(cid, color, dashed):
        v = CECS_DATA[cid]
        values = [v["NI"], v["ND"], v["NDeaf"], v["NFix"]]
        values += values[:1]
        style = "--" if dashed else "-"
        tag = "low CECS" if dashed else "high CECS"
        ax.plot(angles, values, style, color=color, linewidth=2, label=f"Chamber {cid} ({tag})")
        ax.fill(angles, values, color=color, alpha=0.07)

    for cid, color in zip(top, high_colors):
        plot_one(cid, color, dashed=False)
    for cid, color in zip(bottom, low_colors):
        plot_one(cid, color, dashed=True)

    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.5, 1.15), fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_radar.png", bbox_inches="tight")
    plt.close(fig)


def fig6_typology_scatter(chamber_ids):
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    ni = [CECS_DATA[c]["NI"] for c in ids]
    nd = [CECS_DATA[c]["ND"] for c in ids]
    cecs = [CECS_DATA[c]["CECS"] for c in ids]

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    sc = ax.scatter(ni, nd, c=cecs, cmap="RdYlBu_r", s=160, edgecolor="#333333", zorder=3)
    for c, x, y in zip(ids, ni, nd):
        ax.annotate(str(c), (x, y), xytext=(x + 1.3, y + 1.3), fontsize=8.5)
    plt.colorbar(sc, ax=ax, label="CECS")
    ax.set_xlabel("Isolation, $N_I$ (0-100)")
    ax.set_ylabel("Internal Density, $N_D$ (0-100)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_typology.png", bbox_inches="tight")
    plt.close(fig)


def fig10_modularity_diagnostic(chamber_ids):
    """Extra figure (not forced into the paper's original set): shows why
    Modularity Contribution (Q_c) is reported separately rather than folded
    into CECS - it correlates with Isolation/Deafness but isn't one of the
    four scored dimensions. Only runs if Modularity_Score was in the JSON."""
    ids = filter_available(chamber_ids, CECS_DATA, "CECS")
    ids = [c for c in ids if CECS_DATA[c].get("Mod") is not None]
    if len(ids) < 2:
        print("NOTE: not enough Modularity_Score data for Fig. 10 - skipping.")
        return

    mod = np.array([CECS_DATA[c]["Mod"] for c in ids])
    iso = np.array([CECS_DATA[c]["NI"] for c in ids])
    deaf = np.array([CECS_DATA[c]["NDeaf"] for c in ids])
    cecs = np.array([CECS_DATA[c]["CECS"] for c in ids])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8), dpi=200)

    ax1.scatter(iso, mod, c=cecs, cmap="RdYlBu_r", s=110, edgecolor="#333333", zorder=3)
    for c, x, y in zip(ids, iso, mod):
        ax1.annotate(str(c), (x, y), xytext=(x + 1, y + 1), fontsize=8)
    ax1.set_xlabel("Isolation, $N_I$ (0-100)"); ax1.set_ylabel("Modularity Contribution, $Q_c$")
    ax1.set_title("(a) $Q_c$ vs. Isolation", loc="left")

    ax2.scatter(deaf, mod, c=cecs, cmap="RdYlBu_r", s=110, edgecolor="#333333", zorder=3)
    for c, x, y in zip(ids, deaf, mod):
        ax2.annotate(str(c), (x, y), xytext=(x + 1, y + 1), fontsize=8)
    ax2.set_xlabel("Deafness, $N_{Deaf}$ (0-100)"); ax2.set_ylabel("Modularity Contribution, $Q_c$")
    ax2.set_title("(b) $Q_c$ vs. Deafness", loc="left")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig10_modularity_diagnostic.png", bbox_inches="tight")
    plt.close(fig)

    r_iso = np.corrcoef(iso, mod)[0, 1]
    r_deaf = np.corrcoef(deaf, mod)[0, 1]
    print(f"Fig. 10: Pearson r(Q_c, Isolation) = {r_iso:.3f}, r(Q_c, Deafness) = {r_deaf:.3f} (n={len(ids)})")


# =====================================================================
# 4. STANCE FIGURES (Figs. 7-9, use stance_each_community.json)
# =====================================================================

def fig7_coverage_confidence(stance_by_id, stable_id, unstable_id):
    ids = filter_available([stable_id, unstable_id], stance_by_id, "stance")
    if len(ids) < 2:
        print("NOTE: need both example chambers present for Fig. 7 - skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6), dpi=200)
    styles = {stable_id: ("#4C72B0", "-o"), unstable_id: ("#DD8452", "-s")}

    for cid in ids:
        color, style = styles[cid]
        thresholds = stance_by_id[cid]["thresholds"]
        ks = [t["k"] for t in thresholds]
        totals = [t["total_users"] for t in thresholds]
        ax1.plot(ks, totals, style, color=color, label=f"Chamber {cid}", markersize=4)

        dss_vals, dss_ks = [], []
        for t in thresholds:
            dist = t["distribution"]
            a, b = dist.get(STANCE_A, 0), dist.get(STANCE_B, 0)
            if a + b > 0:
                dss_vals.append(max(a, b) / (a + b) * 100)
                dss_ks.append(t["k"])
        ax2.plot(dss_ks, dss_vals, style, color=color, label=f"Chamber {cid}", markersize=4)

    ax1.set_yscale("log")
    ax1.set_xlabel("Evidence threshold k"); ax1.set_ylabel("Classified users remaining")
    ax1.set_title("(a) Coverage vs. k", loc="left"); ax1.legend(fontsize=8, frameon=False)

    ax2.set_xlabel("Evidence threshold k"); ax2.set_ylabel("Dominant Stance Share (%)")
    ax2.set_title("(b) Confidence vs. k", loc="left"); ax2.legend(fontsize=8, frameon=False)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_coverage_confidence.png", bbox_inches="tight")
    plt.close(fig)


def fig8_stance_composition(stance_by_id, chamber_ids, k):
    ids = filter_available(chamber_ids, stance_by_id, "stance")
    rows = [get_stance_row(stance_by_id, c, k) for c in ids]
    rows.sort(key=lambda r: -CECS_DATA.get(r["cid"], {}).get("CECS", 0))

    labels = [str(r["cid"]) for r in rows]
    a = np.array([r["a"] for r in rows]); b = np.array([r["b"] for r in rows])
    gray = np.array([r["gray"] for r in rows]); unclass = np.array([r["unclassified"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=200)
    x = np.arange(len(rows)); width = 0.62
    ax.bar(x, a, width=width, color="#4C72B0", label="Stance A")
    ax.bar(x, b, width=width, bottom=a, color="#DD8452", label="Stance B")
    ax.bar(x, gray, width=width, bottom=a + b, color="#B0B0B0", label="Mixed / gray zone")
    ax.bar(x, unclass, width=width, bottom=a + b + gray, color="#E0E0E0",
           edgecolor="#999999", label="Unclassified")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Chamber ID"); ax.set_ylabel("Share of community users (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_stance_composition.png", bbox_inches="tight")
    plt.close(fig)


def fig9_cecs_vs_dss(stance_by_id, chamber_ids, k):
    ids = filter_available(chamber_ids, stance_by_id, "stance")
    ids = filter_available(ids, CECS_DATA, "CECS")
    rows = [get_stance_row(stance_by_id, c, k) for c in ids]
    rows = [r for r in rows if r["dss"] is not None]

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    for r in rows:
        cecs = CECS_DATA[r["cid"]]["CECS"]
        low_evidence = r["total_users"] > 0 and (r["a"] + r["b"]) < 10   # < ~10% classified
        ax.scatter(cecs, r["dss"], s=90,
                   facecolor="white" if low_evidence else "#4C72B0",
                   edgecolor="#4C72B0", linewidth=1.8, zorder=3)
        ax.annotate(str(r["cid"]), (cecs, r["dss"]), xytext=(cecs + 1.3, r["dss"] + 1.3), fontsize=8.5)

    if len(rows) >= 2:
        xs = np.array([CECS_DATA[r["cid"]]["CECS"] for r in rows])
        ys = np.array([r["dss"] for r in rows])
        coeffs = np.polyfit(xs, ys, 1)
        xline = np.linspace(xs.min() - 2, xs.max() + 2, 50)
        ax.plot(xline, np.polyval(coeffs, xline), "--", color="#999999", linewidth=1.2, zorder=1)
        r_value = np.corrcoef(xs, ys)[0, 1]
        print(f"Fig. 9: Pearson r between CECS and DSS = {r_value:.3f} (n={len(rows)})")

    ax.set_xlabel("CECS (structural echo-chamber score)")
    ax.set_ylabel("Dominant Stance Share, DSS (%)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig9_cecs_vs_dss.png", bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# 5. MAIN
# =====================================================================

def main():
    global CECS_DATA
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Selected chambers: {CHAMBER_IDS}\n")

    load_general_info(GENERAL_JSON)
    CECS_DATA = load_cecs_data(CECS_JSON)
    stance_by_id = load_stance_data(STANCE_JSON)

    fig2_dimensions_bar(CHAMBER_IDS)
    fig3_cecs_ranking(CHAMBER_IDS)
    fig4_correlation_heatmap(CHAMBER_IDS)
    fig5_radar_high_vs_low(CHAMBER_IDS)
    fig6_typology_scatter(CHAMBER_IDS)
    fig10_modularity_diagnostic(CHAMBER_IDS)

    fig7_coverage_confidence(stance_by_id, STABLE_EXAMPLE, UNSTABLE_EXAMPLE)
    fig8_stance_composition(stance_by_id, CHAMBER_IDS, K_THRESHOLD)
    fig9_cecs_vs_dss(stance_by_id, CHAMBER_IDS, K_THRESHOLD)

    print(f"\nDone. Figures saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()