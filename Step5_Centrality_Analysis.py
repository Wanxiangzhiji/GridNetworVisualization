"""SYSE580 | Step 5: Degree, approximate betweenness, and k-core.

Run in VS Code terminal: python Step5_Centrality_Analysis.py
Or in notebook: %run Step5_Centrality_Analysis.py

Requires network.gml/network.gml. Reuses Step4 community CSV if present.
All betweenness values in the default mode are APPROXIMATIONS (sampled sources).
"""
from pathlib import Path
import csv
import time
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt

DATA = Path('network.gml') / 'network.gml'
COMMUNITIES = Path('Step4_Node_Communities.csv')
OUT_CSV = Path('Step5_Node_Centrality.csv')
OUT_PNG = Path('Step5_Centrality_Overview.png')
SEED = 580
BETWEENNESS_SAMPLE_K = 300  # Change to None for exact betweenness (much slower)
TOP_N = 15


def load_graph():
    if not DATA.is_file():
        raise FileNotFoundError(f'Cannot find {DATA.resolve()}')
    raw = DATA.read_text(encoding='utf-8').replace('_pos', 'pos')
    return nx.parse_gml(raw, label='id')


def load_community_ids(G):
    if not COMMUNITIES.is_file():
        print('Step 4 CSV absent: community IDs will be blank.')
        return {}
    with COMMUNITIES.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    found = {int(r['node_id']): int(r['community_id']) for r in rows}
    if set(found) != set(G):
        raise ValueError('Step 4 community file does not match this network.')
    print('Reused community IDs from Step 4.')
    return found


def main():
    G = load_graph()
    print(f'Loaded {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges')
    if G.is_directed():
        raise ValueError('Expected undirected graph.')
    community = load_community_ids(G)
    degree = dict(G.degree())
    kcore = nx.core_number(G)

    k = BETWEENNESS_SAMPLE_K
    if k is not None:
        k = min(k, G.number_of_nodes())
        print(f'Calculating APPROXIMATE betweenness from {k} sampled sources...', flush=True)
    else:
        print('Calculating EXACT betweenness (can take much longer)...', flush=True)
    t0 = time.perf_counter()
    betweenness = nx.betweenness_centrality(G, k=k, normalized=True, seed=SEED)
    print(f'Betweenness finished in {time.perf_counter()-t0:.1f}s')

    ordered = sorted(G.nodes(), key=lambda n: betweenness[n], reverse=True)
    with OUT_CSV.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'community_id', 'degree', 'betweenness_approx' if k is not None else 'betweenness_exact', 'k_core'])
        for n in G:
            writer.writerow([n, community.get(n, ''), degree[n], betweenness[n], kcore[n]])
    print('\nTop nodes by betweenness (structural shortest-path measure):')
    print(f'{"Node":>8} {"Degree":>8} {"Betweenness":>14} {"k-core":>8}')
    for n in ordered[:TOP_N]:
        print(f'{str(n):>8} {degree[n]:>8} {betweenness[n]:>14.6f} {kcore[n]:>8}')
    print('k-core counts:', dict(sorted(Counter(kcore.values()).items())))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    degree_hist = Counter(degree.values())
    xs = sorted(degree_hist)
    ax[0].bar(xs, [degree_hist[x] for x in xs], color='#477aa5')
    ax[0].set(xlabel='Degree', ylabel='Number of nodes', title='Degree distribution')
    ax[0].set_xticks(xs)

    node_order = list(G)
    ax[1].scatter([degree[n] for n in node_order], [betweenness[n] for n in node_order], s=6, alpha=0.35, color='#477aa5')
    ax[1].set(xlabel='Degree', ylabel='Normalized betweenness', title='Degree vs. betweenness')

    core_hist = Counter(kcore.values())
    cs = sorted(core_hist)
    ax[2].bar(cs, [core_hist[x] for x in cs], color='#477aa5')
    ax[2].set(xlabel='Core number', ylabel='Number of nodes', title='k-core distribution')
    ax[2].set_xticks(cs)

    fig.suptitle('Western US Power Grid | Structural centrality (no styling)', fontsize=15)
    fig.text(0.5, 0.01, f'Betweenness: {"approximate, sampled k="+str(k) if k is not None else "exact"}; seed={SEED}. Structural metrics do NOT represent electrical flows or failure risk.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=180, bbox_inches='tight')
    plt.show()
    print('\nSaved:', OUT_CSV.resolve())
    print('Saved:', OUT_PNG.resolve())


if __name__ == '__main__':
    main()
