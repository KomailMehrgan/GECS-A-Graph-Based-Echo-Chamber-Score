"""
GECS — Graph-based Echo Chamber Scoring
=========================================

"""

import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

try:
    import igraph as ig
    import leidenalg
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This script requires python-igraph and leidenalg.\n"
        "Install with: pip install python-igraph leidenalg"
    ) from exc


WEIGHT_ATTR = "weight"


# ---------------------------------------------------------------------------
# 1. Data loading  (same conventions as DATA.py)
# ---------------------------------------------------------------------------

def load_retweet_graph(dataset_name="sxsw", weight_attr=WEIGHT_ATTR):
    """Load the raw retweet graph and (optionally) filter to users present
    in tweets.feather, exactly as DATA.py does. The graph is coerced to be
    directed and every edge is guaranteed to carry a numeric weight
    (defaulting to 1 if the GML file has none), since Eq. 1 requires a
    weighted, directed graph."""
    dataset_path = f"./dataset/{dataset_name}/"
    gml_file = os.path.join(dataset_path, "graph.gml")
    tweets_file = os.path.join(dataset_path, "tweets.feather")

    print(f"Loading raw retweet graph from '{gml_file}'...")
    G = nx.read_gml(gml_file)
    if not G.is_directed():
        G = G.to_directed()

    if os.path.exists(tweets_file):
        df = pd.read_feather(tweets_file)
        G = G.subgraph(df["user_id"])

    # Ensure every edge has a numeric weight (retweet count).
    for u, v, data in G.edges(data=True):
        if weight_attr not in data:
            data[weight_attr] = 1.0
        else:
            data[weight_attr] = float(data[weight_attr])

    return G


def extract_largest_component(G):
    """Extract the largest connected component (weakly connected, since G
    is directed) and relabel nodes to clean integer indices, matching the
    LCC convention used elsewhere in the codebase."""
    lcc_nodes = max(nx.weakly_connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes).copy()

    node_id_map = {node: i for i, node in enumerate(G_lcc.nodes())}
    G_lcc = nx.relabel_nodes(G_lcc, node_id_map)

    print(
        f"LCC ready: {G_lcc.number_of_nodes()} nodes, "
        f"{G_lcc.number_of_edges()} edges."
    )
    return G_lcc


# ---------------------------------------------------------------------------
# 2. Community detection via Leiden
# ---------------------------------------------------------------------------

def detect_leiden_communities(G, resolution=0.3, seed=42, weight_attr=WEIGHT_ATTR):
    """Partition G with the Leiden algorithm (RBConfiguration objective,
    i.e. Newman's modularity generalized with resolution parameter gamma,
    Eq. 2). Returns a list of sets of original G node ids."""
    print(f"Running Leiden community detection (resolution={resolution})...")

    nodes = list(G.nodes())
    node_index = {n: i for i, n in enumerate(nodes)}

    edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
    weights = [G[u][v][weight_attr] for u, v in G.edges()]

    g_ig = ig.Graph(n=len(nodes), edges=edges, directed=True)
    g_ig.es[weight_attr] = weights

    partition = leidenalg.find_partition(
        g_ig,
        leidenalg.RBConfigurationVertexPartition,
        weights=weight_attr,
        resolution_parameter=resolution,
        seed=seed,
    )

    communities = []
    for cluster in partition:
        communities.append({nodes[i] for i in cluster})

    print(f"Leiden found {len(communities)} communities.")
    return communities


# ---------------------------------------------------------------------------
# 3. Boundary weights & the (diagnostic-only) modularity contribution Q_c
# ---------------------------------------------------------------------------

def compute_global_weight(G, weight_attr=WEIGHT_ATTR):
    """W_global: total weight of the whole network."""
    return sum(data.get(weight_attr, 1.0) for _, _, data in G.edges(data=True))


def compute_boundary_weights(G, community, weight_attr=WEIGHT_ATTR):
    """W_int: weight of edges with both endpoints inside the community.
    W_ext: weight of edges with exactly one endpoint inside the community.
    Both directions of a directed edge are counted, since interaction
    weight (not degree) is what matters for Eqs. 4-6."""
    community_set = set(community)
    w_int, w_ext = 0.0, 0.0

    for u, v, data in G.edges(data=True):
        w = data.get(weight_attr, 1.0)
        u_in, v_in = u in community_set, v in community_set
        if u_in and v_in:
            w_int += w
        elif u_in or v_in:
            w_ext += w

    return w_int, w_ext


def compute_qc(w_int, w_total, w_global):
    """Per-community modularity contribution (Eq. 3). Reported as a
    diagnostic in Section IV-C style analysis only — NOT part of GECS,
    per the paper's three-point justification for excluding it."""
    if w_global == 0:
        return 0.0
    return (w_int / w_global) - (w_total / (2 * w_global)) ** 2


# ---------------------------------------------------------------------------
# 4. The four GECS dimensions
# ---------------------------------------------------------------------------

def isolation_metric(w_int, w_ext):
    """Isolation via the normalized Isolation Metric (Eq. 4, Eq. 7).
    IM is retained as the single boundary-cohesion representative since
    IM, E/I, and Phi are monotone transforms of one another."""
    denom = w_int + w_ext
    if denom == 0:
        return 0.0, 0.0
    IM = w_int / denom
    return IM, IM * 100.0


def internal_density(G, community, upper_bound=0.4):
    """Internal Density via average local clustering coefficient
    (Eq. 10), normalized against an empirical upper bound of 0.70
    (Eq. 11). Clustering is computed on the undirected projection of the
    community subgraph, matching the definition C_i = 2e_i / (k_i(k_i-1))."""
    sub = G.subgraph(community)
    sub_u = sub.to_undirected() if sub.is_directed() else sub

    if sub_u.number_of_nodes() < 3:
        return 0.0, 0.0

    clustering = nx.clustering(sub_u)
    C = float(np.mean(list(clustering.values())))
    N_D = min((C / upper_bound) * 100.0, 100.0)
    return C, N_D


def deafness_score(G, community, degree_centrality):
    """Diplomat Index / Deafness (Eqs. 12-14).

    Diplomats: community members with at least one edge (in or out)
    crossing the community boundary.
    DE: the diplomats' SHARE of the community's total degree centrality
    mass, i.e. sum(centrality of diplomats) / sum(centrality of all
    community members). Since diplomats are a subset of all members and
    centrality is non-negative, this ratio is naturally bounded in
    [0, 1] — unlike a ratio of means, which can exceed 1 whenever the
    bridging users are (as is typical) the community's highest-degree
    hubs, driving Deafness negative.
    Deafness = 1 - DE. A community with NO diplomats is, by definition,
    fully sealed off, so we assign it maximal Deafness (100)."""
    community_set = set(community)

    diplomats = [
        n for n in community_set
        if any(
            nbr not in community_set
            for nbr in set(G.successors(n)) | set(G.predecessors(n))
        )
    ]

    if not diplomats:
        return 1.0, 100.0

    dc_diplomats = sum(degree_centrality[n] for n in diplomats)
    dc_all = sum(degree_centrality[n] for n in community_set)

    if dc_all == 0:
        return 1.0, 100.0

    DE = dc_diplomats / dc_all  # naturally in [0, 1]
    Deafness = 1.0 - DE
    return Deafness, Deafness * 100.0


def fixation_score(G, community, node_to_comm, weight_attr=WEIGHT_ATTR, entropy_cap=1.05):
    """Neighborhood Entropy / Fixation (Eq. 15-16).

    For every node in the community that has at least one external tie,
    build its distribution of external interaction weight across
    neighboring communities, then take the Shannon entropy of that
    distribution. H_c is the average per-node entropy over those nodes
    ONLY.

    Nodes with zero external interaction are EXCLUDED from the average
    (not scored as entropy 0): whether a node has any external ties at
    all is already what Isolation measures. Folding "no external ties"
    into Fixation as an entropy-0 data point would make Fixation
    redundant with Isolation instead of measuring a distinct question —
    given that external attention exists, is it concentrated or spread
    out? If a community has literally no external ties anywhere,
    Fixation is undefined for it and we return N_Fix = 0.0 (no
    concentration signal to report; the community's Isolation score
    already tells that story)."""
    community_set = set(community)
    entropies = []

    for node in community_set:
        ext_weights = {}
        neighbors = set(G.successors(node)) | set(G.predecessors(node))

        for nbr in neighbors:
            if nbr in community_set:
                continue
            nbr_comm = node_to_comm.get(nbr)
            if nbr_comm is None:
                continue  # neighbor filtered out (e.g. tiny community dropped)

            w = 0.0
            if G.has_edge(node, nbr):
                w += G[node][nbr].get(weight_attr, 1.0)
            if G.has_edge(nbr, node):
                w += G[nbr][node].get(weight_attr, 1.0)

            ext_weights[nbr_comm] = ext_weights.get(nbr_comm, 0.0) + w

        total = sum(ext_weights.values())
        if total == 0:
            continue  # no external ties: excluded, not scored as entropy 0

        h = 0.0
        for w in ext_weights.values():
            p = w / total
            if p > 0:
                h -= p * np.log2(p)
        entropies.append(h)

    if not entropies:
        return 0.0, 0.0  # no external interaction anywhere: Fixation undefined

    H_c = float(np.mean(entropies))
    N_Fix = (1.0 - H_c / entropy_cap) * 100.0
    N_Fix = float(np.clip(N_Fix, 0.0, 100.0))
    return H_c, N_Fix


# ---------------------------------------------------------------------------
# 5. Composite GECS score
# ---------------------------------------------------------------------------

def compute_gecs(N_I, N_D, N_Deaf, N_Fix, weights=(0.375, 0.25, 0.25, 0.125)):
    """Composite score (Eq. 17). Default weights give Isolation the
    highest weight, per the paper's statement that Isolation is the
    'defining characteristic of an echo chamber.' Adjust as needed, but
    keep the weights summing to 1 so GECS stays bounded on [0, 100]."""
    w1, w2, w3, w4 = weights
    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"GECS weights must sum to 1.0, got {sum(weights)}")
    return w1 * N_I + w2 * N_D + w3 * N_Deaf + w4 * N_Fix


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

def analyze_dataset(
    dataset_name="sxsw",
    leiden_resolution=1,
    gecs_weights=(0.375, 0.25, 0.25, 0.125),
    min_community_size=100,
):
    """Run the full GECS pipeline end-to-end and return everything needed
    for reporting: the LCC graph, the retained communities, a node->community
    lookup, and a results DataFrame sorted by GECS descending."""
    G = load_retweet_graph(dataset_name)
    G = extract_largest_component(G)

    raw_communities = detect_leiden_communities(G, resolution=leiden_resolution)

    communities = [c for c in raw_communities if len(c) >= min_community_size]
    dropped = len(raw_communities) - len(communities)
    if dropped:
        print(f"Dropped {dropped} communities smaller than min_community_size={min_community_size}.")

    node_to_comm = {}
    for cid, community in enumerate(communities):
        for node in community:
            node_to_comm[node] = cid

    W_global = compute_global_weight(G)
    degree_centrality = nx.degree_centrality(G)

    rows = []
    print("Scoring communities...")
    for cid, community in enumerate(communities):
        w_int, w_ext = compute_boundary_weights(G, community)
        w_total = w_int + w_ext

        IM, N_I = isolation_metric(w_int, w_ext)
        C, N_D = internal_density(G, community)
        Deafness, N_Deaf = deafness_score(G, community, degree_centrality)
        H_c, N_Fix = fixation_score(G, community, node_to_comm)
        Q_c = compute_qc(w_int, w_total, W_global)
        GECS = compute_gecs(N_I, N_D, N_Deaf, N_Fix, gecs_weights)

        rows.append({
            "community_id": cid,
            "size": len(community),
            "W_int": w_int,
            "W_ext": w_ext,
            "W_total": w_total,
            "IM": IM,
            "N_Isolation": N_I,
            "clustering_coef": C,
            "N_Density": N_D,
            "Deafness": Deafness,
            "N_Deafness": N_Deaf,
            "H_c": H_c,
            "N_Fixation": N_Fix,
            "Q_c_diagnostic": Q_c,
            "GECS": GECS,
        })

    df = pd.DataFrame(rows).sort_values("GECS", ascending=False).reset_index(drop=True)
    return G, communities, node_to_comm, df


# ---------------------------------------------------------------------------
# 7. Reporting & visualization
# ---------------------------------------------------------------------------

def save_report(df, dataset_name, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{dataset_name}_gecs_scores.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved scores to '{csv_path}'.")

    print("\n=== GECS Summary ===")
    print(df[[
        "community_id", "size", "N_Isolation", "N_Density",
        "N_Deafness", "N_Fixation", "GECS", "Q_c_diagnostic",
    ]].to_string(index=False))

    return csv_path


def plot_gecs_scores(df, dataset_name, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ordered = df.sort_values("GECS", ascending=True)
    bars = ax.barh(
        ordered["community_id"].astype(str),
        ordered["GECS"],
        color=plt.cm.Reds(ordered["GECS"] / 100.0),
    )
    ax.set_xlabel("GECS (0-100)")
    ax.set_ylabel("Community ID")
    ax.set_title(f"GECS Echo-Chamber Score by Community — '{dataset_name}'")
    ax.set_xlim(0, 100)
    ax.bar_label(bars, fmt="%.1f", padding=3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset_name}_gecs_scores.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved GECS bar chart to '{out_path}'.")
    return out_path


def plot_dimension_breakdown(df, dataset_name, out_dir="output"):
    """Stacked/side-by-side view of the four normalized dimensions per
    community, so the composite score's composition is inspectable."""
    os.makedirs(out_dir, exist_ok=True)
    dims = ["N_Isolation", "N_Density", "N_Deafness", "N_Fixation"]
    labels = ["Isolation", "Density", "Deafness", "Fixation"]

    ordered = df.sort_values("GECS", ascending=False)
    x = np.arange(len(ordered))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.8), 6))
    for i, (dim, label) in enumerate(zip(dims, labels)):
        ax.bar(x + i * width, ordered[dim], width, label=label)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(ordered["community_id"].astype(str))
    ax.set_ylabel("Normalized score (0-100)")
    ax.set_xlabel("Community ID")
    ax.set_title(f"GECS Dimension Breakdown — '{dataset_name}'")
    ax.legend()

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{dataset_name}_gecs_dimensions.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved dimension breakdown chart to '{out_path}'.")
    return out_path


def plot_community_graph(G, node_to_comm, dataset_name, out_dir="output"):
    """Spring-layout visualization of the LCC colored by Leiden community,
    in the same visual style as visualize_dataset_graph() in DATA.py."""
    os.makedirs(out_dir, exist_ok=True)

    scored_nodes = [n for n in G.nodes() if n in node_to_comm]
    G_scored = G.subgraph(scored_nodes)
    labels = [node_to_comm[n] for n in G_scored.nodes()]

    print("Calculating spring layout...")
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G_scored, seed=42, k=0.15)

    nx.draw_networkx_nodes(
        G_scored, pos,
        node_size=15,
        node_color=labels,
        cmap=plt.cm.Set1,
        alpha=0.8,
    )
    nx.draw_networkx_edges(G_scored, pos, alpha=0.05, edge_color="gray")

    plt.title(
        f"Retweet Graph Colored by Leiden Community: '{dataset_name.capitalize()}'",
        fontsize=18, fontweight="bold",
    )
    plt.axis("off")
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{dataset_name}_community_graph.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved community graph to '{out_path}'.")
    return out_path


# ---------------------------------------------------------------------------
# 8. CLI entry point (Processes all four datasets and compiles result.txt)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the GECS echo-chamber scoring pipeline across all datasets.")
    parser.add_argument("--datasets", nargs="+", default=["gun", "super_bowl", "sxsw", "abortion" , "beefban_full","beefban_threshold","gunsense_threshold","nationalkissingday_threshold","ukraine_full","ukraine_threshold","ultralive_threshold"],
                        help="List of dataset folder names under data/")
    parser.add_argument("--resolution", type=float, default=0.3, help="Leiden resolution parameter (gamma)")
    parser.add_argument("--min-size", type=int, default=100, help="Minimum community size to score")
    parser.add_argument("--weights", type=float, nargs=4, default=[0.375, 0.25, 0.25, 0.125],
                         metavar=("W_ISOLATION", "W_DENSITY", "W_DEAFNESS", "W_FIXATION"),
                         help="Composite weights, must sum to 1.0")
    parser.add_argument("--out-dir", default="output", help="Directory for CSVs, plots, and result.txt")
    parser.add_argument("--no-plots", action="store_false", help="Skip plot generation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    result_txt_path = os.path.join(args.out_dir, "result.txt")

    with open(result_txt_path, "w", encoding="utf-8") as f_all:
        f_all.write("=== GECS Analysis Results for All Datasets ===\n\n")

        for dataset_name in args.datasets:
            print(f"\n{'='*40}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*40}")

            f_all.write(f"Dataset: {dataset_name}\n")
            f_all.write("-" * 40 + "\n")

            try:
                G, communities, node_to_comm, df = analyze_dataset(
                    dataset_name=dataset_name,
                    leiden_resolution=args.resolution,
                    gecs_weights=tuple(args.weights),
                    min_community_size=args.min_size,
                )

                csv_path = save_report(df, dataset_name, out_dir=args.out_dir)

                # Append summary table and CSV path to result.txt
                f_all.write(f"CSV Report: {csv_path}\n\n")
                f_all.write(df[[
                    "community_id", "size", "N_Isolation", "N_Density",
                    "N_Deafness", "N_Fixation", "GECS", "Q_c_diagnostic",
                ]].to_string(index=False))
                f_all.write("\n\n" + "="*50 + "\n\n")

                if not args.no_plots:
                    plot_gecs_scores(df, dataset_name, out_dir=args.out_dir)
                    plot_dimension_breakdown(df, dataset_name, out_dir=args.out_dir)
                    plot_community_graph(G, node_to_comm, dataset_name, out_dir=args.out_dir)

            except Exception as e:
                err_msg = f"Error processing dataset '{dataset_name}': {str(e)}"
                print(err_msg)
                f_all.write(f"{err_msg}\n\n" + "="*50 + "\n\n")

    print(f"\nAll datasets processed successfully. Combined results saved to '{result_txt_path}'.")


if __name__ == "__main__":
    main()