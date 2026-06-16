"""
Logic: 

CLUSTER-BIASED node2vec on a Planetoid graph

--------
node2vec's walk picks the next node x from current node v using:

        pi(v -> x) = ß(t, x) * w(v, x)

ß_pq depends on the previous node t (the p/q machinery).
w(v, x) is just the edge weight (1 for an unweighted graph like Cora).

This cluster bias beta(v, x) depends only on v and x (are they in the same
cluster? / how deep do they share a cluster?). It does NOT depend on t.

Because of that, we can fold beta straight into the edge weight:

        w'(v, x) = a(v, x) * w(v, x)

Then, run ordinary node2vec. The walk now:

        pi(v -> x) = a_pq(t, x) * ß(v, x) * w(v, x)

A cluster rule with NO rewrite of the library's internals. We still walk only on real edges; we just reweight them.

Two modes
---------
  flat       : ß = ß_in if v,x share the finest Louvain community,
               else beta_out.   (one simple knob)

  multiscale : beta interpolates by How Deep v,x share a cluster in the
               Louvain dendrogram (deep shared cluster -> closer to beta_in,
               only-share-the-root -> closer to beta_out).   (the "hierarchical"
               / multi-scale version your topic title asks for)

"""

import argparse
import os
import random
import warnings

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

import community as community_louvain          # python-louvain
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")




# Reproducibility 
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# Load Data 
def load_planetoid(name, root="data"):
    data = Planetoid(root=os.path.join(root, name), name=name)[0]
    G = to_networkx(data, to_undirected=True)
    y = data.y.numpy()
    print(f"[data] {name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"{int(y.max()) + 1} classes")
    return G, y, data.train_mask.numpy(), data.test_mask.numpy()


# Train a simple classifier for evaluation
def evaluate(X, y, train_mask, test_mask, seed):
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[train_mask], y[train_mask])
    return accuracy_score(y[test_mask], clf.predict(X[test_mask]))


def embeddings_to_matrix(model, N, d):
    X = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        if str(i) in model.wv:
            X[i] = model.wv[str(i)]
    return X


def plot_tsne(X, y, title, path, seed):
    coords = TSNE(n_components=2, init="pca", learning_rate="auto",
                  random_state=seed).fit_transform(X)
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(coords[:, 0], coords[:, 1], c=y, s=8, cmap="tab10", alpha=0.8)
    plt.legend(*sc.legend_elements(), title="class", loc="best", fontsize=8)
    plt.title(title); plt.xticks([]); plt.yticks([]); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"[tsne] saved -> {path}")


# 1) Hierarchical clustering (Louvain dendrogram)

# Return a list `level_maps` of {node -> community} dicts, one per level, from Finest (index 0) to Coarest (last index).
def compute_louvain_levels(G, seed):
    dendro = community_louvain.generate_dendrogram(G, random_state=seed)
    level_maps = [community_louvain.partition_at_level(dendro, lvl)
                  for lvl in range(len(dendro))]
    sizes = [len(set(m.values())) for m in level_maps]
    print(f"[louvain] {len(level_maps)} levels; communities per level "
          f"(fine->coarse): {sizes}")
    return level_maps



"""
    This function return a function beta(u, v) used to reweight edge (u, v).
    flat        -> beta_in if u,v in the same finest community else beta_out.
    multiscale  -> geometric interpolation by how deep u,v first share a
                   community: deep (fine level) -> beta_in, shallow/root -> beta_out.
"""
def make_beta_fn(level_maps, beta_in, beta_out, mode):
    H = len(level_maps)

    def beta(u, v):
        if mode == "flat":
            same = level_maps[0][u] == level_maps[0][v]
            return beta_in if same else beta_out

        # multiscale: find finest level where u, v share a community
        merge_level = None
        for lvl in range(H):
            if level_maps[lvl][u] == level_maps[lvl][v]:
                merge_level = lvl
                break
        if merge_level is None:          # never share -> only meet at the root
            s = 0.0
        else:                            # fine merge (small level) -> s near 1
            s = (H - merge_level) / H
        return beta_out * (beta_in / beta_out) ** s

    return beta


# 2) The cluster-biased node2vec
class ClusterBiasedNode2Vec(Node2Vec):
    """
    node2vec whose graph edges have been reweighted by a cluster-bias factor
    ß(u, v) BEFORE the walks are generated. Everything else (p, q, the walk
    generation, the Skip-gram training) is exactly as it is in node2vec.
    """

    def __init__(self, graph, level_maps, beta_in=2.0, beta_out=0.5,
                 mode="multiscale", weight_key="weight", **kwargs):
        beta = make_beta_fn(level_maps, beta_in, beta_out, mode)

        # Build a reweighted copy of the graph (don't touch the original)
        biased = graph.copy()
        for u, v, d in biased.edges(data=True):
            base_w = d.get(weight_key, 1.0)
            d[weight_key] = base_w * beta(u, v)

        # Hand the reweighted graph (biased = graph.copy()) to the normal node2vec
        super().__init__(biased, weight_key=weight_key, **kwargs)


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def fit_and_eval(n2v_obj, G, y, train_mask, test_mask, dims, window, seed, label):
    model = n2v_obj.fit(window=window, min_count=1, sg=1, seed=seed, workers=1)
    X = embeddings_to_matrix(model, G.number_of_nodes(), dims)
    acc = evaluate(X, y, train_mask, test_mask, seed)
    print(f"[eval] {label:<34} test acc = {acc:.4f}")
    return acc, X


def main():
    ap = argparse.ArgumentParser(description="Idea A: cluster-biased node2vec")
    ap.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    ap.add_argument("--dimensions", type=int, default=128)
    ap.add_argument("--walk_length", type=int, default=20)
    ap.add_argument("--num_walks", type=int, default=10)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--p", type=float, default=1.0)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    G, y, train_mask, test_mask = load_planetoid(args.dataset)
    level_maps = compute_louvain_levels(G, args.seed)

    common = dict(dimensions=args.dimensions, walk_length=args.walk_length,
                  num_walks=args.num_walks, p=args.p, q=args.q,
                  workers=1, seed=args.seed, quiet=True)

    results, best = [], {"acc": -1, "label": None, "X": None}

    # (0) plain node2vec baseline
    print("\n=== baseline node2vec (no cluster bias) ===")
    base = Node2Vec(G, **common)
    acc, X = fit_and_eval(base, G, y, train_mask, test_mask,
                          args.dimensions, args.window, args.seed,
                          "baseline node2vec")
    results.append(("baseline", "-", "-", acc))
    best = {"acc": acc, "label": "baseline", "X": X}

    # (1) cluster-biased variants 
    
    configs = [
        ("flat",       1.25, 1.0),   # gentle: only prompt the walk to stay
        ("multiscale", 1.5,  1.0),   # gentle, multi-scale (deep cluster -> stronger)
        ("flat",       2.0,  0.5),   # aggressive: over-confines -> usually worse
    ]
    for mode, b_in, b_out in configs:

        label = f"cluster-{mode} (in={b_in}, out={b_out})"

        print(f"\n=== {label} ===")
        cb = ClusterBiasedNode2Vec(G, level_maps, beta_in=b_in, beta_out=b_out,
                                   mode=mode, **common)
        acc, X = fit_and_eval(cb, G, y, train_mask, test_mask,
                              args.dimensions, args.window, args.seed, label)
        results.append((f"cluster-{mode}", b_in, b_out, acc))
        if acc > best["acc"]:
            best = {"acc": acc, "label": label, "X": X}

    # summary in a table
    print("\n================ SUMMARY (" + args.dataset + ") ================")
    print(f"{'method':<18}{'beta_in':>9}{'beta_out':>10}{'test_acc':>11}")
    for method, bi, bo, acc in results:
        print(f"{method:<18}{str(bi):>9}{str(bo):>10}{acc:>11.4f}")

    csv_path = os.path.join(args.outdir, f"{args.dataset}_idea_a.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,method,beta_in,beta_out,test_accuracy\n")
        for method, bi, bo, acc in results:
            f.write(f"{args.dataset},{method},{bi},{bo},{acc:.4f}\n")
    print(f"\n[save] results -> {csv_path}")

    tsne_path = os.path.join(args.outdir, f"{args.dataset}_idea_a_best.png")
    plot_tsne(best["X"], y,
              title=f"{args.dataset} {best['label']} (acc={best['acc']:.3f})",
              path=tsne_path, seed=args.seed)

    print(f"\nDone. Best on {args.dataset}: {best['label']} "
          f"= {best['acc']:.4f}")


if __name__ == "__main__":
    main()
