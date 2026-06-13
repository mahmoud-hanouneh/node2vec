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

from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")  # silence sklearn convergence chatter for a clean log


# Reproducibility 

def set_seed(seed: int) -> None:
    """Pin every RNG we touch so runs are comparable across the q-sweep."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# Load Data 

def load_planetoid(name: str, root: str = "data"):
    dataset = Planetoid(root=os.path.join(root, name), name=name)
    data = dataset[0]

    G = to_networkx(data, to_undirected=True) # undirected NetworkX graph, nodes labelled 0..N-1

    y = data.y.numpy() # (N, ) int array of class labels
    
    # bool arrays (N, )
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()

    print(f"[data] {name}: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges, {int(y.max()) + 1} classes")
    print(f"[data] train nodes: {train_mask.sum()}, test nodes: {test_mask.sum()}")
    return G, y, train_mask, test_mask


def run_node2vec(G, p, q, dimensions, walk_length, num_walks, window, seed, workers=1):
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True
    )

    model = node2vec.fit(window=window, min_count=1, sg=1, seed=seed, workers=workers)

    N = G.number_of_nodes()
    X = np.zeros((N, dimensions), dtype=np.float32)
    missing = 0
    for i in range(N):
        key = str(i)
        if key in model.wv:
            X[i] = model.wv[key]
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing} nodes had no embedding (isolated); set to zeros")
    return X


# Train a simple classifier for evaluation
def evaluate(X, y, train_mask, test_mask, seed):
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X[train_mask], y[train_mask])
    pred = clf.predict(X[test_mask])
    return accuracy_score(y[test_mask], pred)


# Visualisation - t-distributed stochastic neighbor embeddings
def plot_tsne(X, y, title, path, seed):
    print(f"[tsne] projecting {X.shape[0]} points -> 2-D ...")
    coords = TSNE(n_components=2, init="pca", learning_rate="auto",
                  random_state=seed).fit_transform(X)

    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=y, s=8,
                          cmap="tab10", alpha=0.8)
    plt.legend(*scatter.legend_elements(), title="class",
               loc="best", fontsize=8)
    plt.title(title)
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[tsne] saved -> {path}")

def main():
    parser = argparse.ArgumentParser(description="node2vec baseline on Planetoid Cora")
    parser.add_argument("--dataset", default="Cora", choices=["Cora", "CiteSeer", "PubMed"])
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--walk_length", type=int, default=20)
    parser.add_argument("--num_walks", type=int, default=10)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q_grid", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    G, y, train_mask, test_mask = load_planetoid(args.dataset)

    results = []
    best = {"acc": -1.0, "q": None, "X": None}

    for q in args.q_grid:
        print(f"\n=== node2vec  p={args.p}  q={q} ===")
        X = run_node2vec(
            G, p=args.p, q=q,
            dimensions=args.dimensions,
            walk_length=args.walk_length,
            num_walks=args.num_walks,
            window=args.window,
            seed=args.seed,
        )
        acc = evaluate(X, y, train_mask, test_mask, args.seed)
        tag = " (= DeepWalk)" if q == 1.0 and args.p == 1.0 else ""
        print(f"[eval] test accuracy = {acc:.4f}{tag}")
        results.append((args.p, q, acc))
        if acc > best["acc"]:
            best = {"acc": acc, "q": q, "X": X}

    # Results 
    print("\n================ SUMMARY (" + args.dataset + ") ================")
    print(f"{'p':>5} {'q':>6} {'test_acc':>10}")
    for p, q, acc in results:
        star = "  <-- best" if q == best["q"] else ""
        print(f"{p:>5.2f} {q:>6.2f} {acc:>10.4f}{star}")

    csv_path = os.path.join(args.outdir, f"{args.dataset}_node2vec_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("dataset,p,q,test_accuracy\n")
        for p, q, acc in results:
            f.write(f"{args.dataset},{p},{q},{acc:.4f}\n")
    print(f"\n[save] results table -> {csv_path}")

    # ---- t-SNE of the best embedding ----
    tsne_path = os.path.join(args.outdir, f"{args.dataset}_tsne_best_q{best['q']}.png")
    plot_tsne(best["X"], y,
              title=f"{args.dataset} node2vec embedding (p={args.p}, q={best['q']}, "
                    f"acc={best['acc']:.3f})",
              path=tsne_path, seed=args.seed)

    print("\nDone. Headline number: "
          f"{args.dataset} best test accuracy = {best['acc']:.4f} at q={best['q']}.")
    

if __name__ == "__main__":
        main()