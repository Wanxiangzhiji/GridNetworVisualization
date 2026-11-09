"""SYSE580 | Step 4: Louvain communities on Western US Power Grid.

Run from your VisualProject directory (VS Code / Jupyter %run).
Dependencies: networkx>=3.1, numpy, matplotlib.
"""
from pathlib import Path
from collections import Counter
import csv
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Your local directory: VisualProject/network.gml/network.gml
DATA_PATH = Path('network.gml') / 'network.gml'
OUTPUT_FIG = Path('Step4_Louvain_Communities.png')
OUTPUT_CSV = Path('Step4_Node_Communities.csv')
LAYOUT_CACHE = Path('Step4_Spring_Positions.csv')
SEED = 580
SPRING_ITERATIONS = 45  # same settings as Step 3
HIGHLIGHT_TOP = 12      # display choice only; all communities retained


def load_graph():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f'Cannot find {DATA_PATH.resolve()}')
    text = DATA_PATH.read_text(encoding='utf-8').replace('_pos', 'pos')
    G = nx.parse_gml(text, label='id')
    orig = {}
    for n, attrs in G.nodes(data=True):
        x, y = map(float, attrs['pos'].split(','))
        orig[n] = np.array([x, y], dtype=float)
    return G, orig


def normalize(pos):
    ids = list(pos)
    xy = np.asarray([pos[n] for n in ids], dtype=float)
    xy -= xy.mean(axis=0)
    xy /= max(np.max(np.abs(xy)), 1e-12)
    return dict(zip(ids, xy))


def get_spring_positions(G, original):
    if LAYOUT_CACHE.exists():
        try:
            with LAYOUT_CACHE.open('r', encoding='utf-8', newline='') as f:
                rows = list(csv.DictReader(f))
            raw = {int(r['node_id']): np.array([float(r['x']), float(r['y'])]) for r in rows}
            if set(raw) == set(G):
                print('Reusing cached Spring coordinates.')
                return raw
        except (ValueError, KeyError, OSError):
            print('Layout cache invalid; recalculating.')
    print('Calculating Spring layout (45 iterations)...', flush=True)
    t0 = time.perf_counter()
    raw = nx.spring_layout(G, pos=normalize(original), seed=SEED,
                           iterations=SPRING_ITERATIONS, scale=1.0)
    raw = normalize(raw)
    with LAYOUT_CACHE.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['node_id', 'x', 'y'])
        for node in G:
            writer.writerow([node, float(raw[node][0]), float(raw[node][1])])
    print(f'Spring layout finished in {time.perf_counter()-t0:.1f}s; cached.')
    return raw


def main():
    G, orig = load_graph()
    print(f'Loaded {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges')
    communities = nx.community.louvain_communities(G, seed=SEED, resolution=1.0)
    communities = sorted(communities, key=len, reverse=True)
    membership = {n: i+1 for i, group in enumerate(communities) for n in group}
    sizes = [len(group) for group in communities]
    modularity = nx.community.modularity(G, communities)
    print(f'Louvain communities: {len(communities)}')
    print(f'Modularity Q: {modularity:.4f}')
    print('Largest 12 community sizes:', sizes[:12])

    with OUTPUT_CSV.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'community_id', 'degree'])
        for n in G:
            w.writerow([n, membership[n], G.degree(n)])
    print('Saved node community assignments:', OUTPUT_CSV)

    pos = get_spring_positions(G, orig)
    # Top 12 communities have individual colors; all others share gray.
    # This is only a display simplification, not a merger in the analysis.
    palette = plt.colormaps['tab20'].colors
    colors = {cid: palette[(cid-1)*2 % len(palette)] for cid in range(1, HIGHLIGHT_TOP+1)}
    node_list = list(G)
    node_colors = [colors.get(membership[n], '#a9aeb9') for n in node_list]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 8),
                                gridspec_kw={'width_ratios': [2.15, 1]})
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.35, alpha=0.19, edge_color='#596579')
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors,
                           node_size=3.3, linewidths=0, ax=ax, alpha=0.94)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    ax.set_title('Louvain communities on Spring layout', fontsize=15, pad=15)
    ax.text(0.01, 0.02,
            f'Top {HIGHLIGHT_TOP} communities colored; remaining {len(communities)-HIGHLIGHT_TOP} in gray',
            transform=ax.transAxes, fontsize=10, color='#536273')

    count = min(20, len(communities))
    ids = np.arange(1, count+1)
    bar_colors = [colors.get(i, '#a9aeb9') for i in ids]
    ax2.barh(ids, sizes[:count], color=bar_colors)
    ax2.invert_yaxis()
    ax2.set_yticks(ids)
    ax2.set_yticklabels([f'C{i}' for i in ids], fontsize=9)
    ax2.set_xlabel('Number of nodes')
    ax2.set_title(f'Top {count} community sizes', fontsize=15, pad=15)
    ax2.grid(axis='x', alpha=0.15)
    ax2.set_axisbelow(True)
    ax2.spines[['right', 'top']].set_visible(False)

    fig.suptitle(f'Western US Power Grid  |  {len(communities)} communities  |  '
                 f'Modularity Q = {modularity:.3f}', fontsize=18, y=0.98)
    fig.text(0.5, 0.025,
             'Real network: 4,941 nodes / 6,594 edges. Community detection: Louvain (seed=580, resolution=1). '
             'Colors show algorithmic groups, not geographic or operational regions.',
             ha='center', fontsize=9, color='#586475')
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.09, wspace=0.12)
    fig.savefig(OUTPUT_FIG, dpi=180, bbox_inches='tight')
    plt.show()
    print('Saved figure:', OUTPUT_FIG.resolve())


if __name__ == '__main__':
    main()
