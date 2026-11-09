"""SYSE580 Step 6.4 | Real 3D spring layout initialized from the saved 2D spring layout.
Run in the VisualProject working directory (VS Code Terminal or %run).
Requires: networkx >= 3.4, numpy, pandas, scipy, plotly.
No synthetic edges, power-flow values, or geographic coordinates are introduced.
"""
from pathlib import Path
import colorsys
import time
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# ---- Configuration: change only after the first run ----
DATA = Path('network.gml') / 'network.gml'
METRICS = Path('Step5_Node_Centrality.csv')
XY_FILE = Path('Step4_Spring_Positions.csv')
CACHE = Path('Step6_4_Positions_3D_spring.csv')
HTML = Path('Step6_4_Interactive_3D_Spring.html')
SEED = 580
ITERATIONS = 35  # Start modestly; 3D spring on ~5000 nodes is computationally expensive
Z_STRENGTH = 0.22  # Initial depth compared with normalized x/y (not physical elevation)
RECOMPUTE = False  # True forces a new layout; otherwise reads existing CACHE

# ---- Load and validate ----
for path in (DATA, METRICS, XY_FILE):
    if not path.is_file():
        raise FileNotFoundError(f'Missing: {path.resolve()}')
raw = DATA.read_text(encoding='utf-8')
G = nx.parse_gml(raw.replace('_pos', 'pos'), label='id')
if G.is_directed():
    raise ValueError('Expected undirected grid data')
nodes = list(G.nodes())
node_set = set(nodes)
metrics = pd.read_csv(METRICS).set_index('node_id')
xy_df = pd.read_csv(XY_FILE).set_index('node_id')
if set(metrics.index) != node_set or set(xy_df.index) != node_set:
    raise ValueError('Node IDs differ across GML, metrics CSV, or 2D positions CSV')
print(f'Loaded {len(nodes):,} nodes and {G.number_of_edges():,} edges', flush=True)

if CACHE.is_file() and not RECOMPUTE:
    cache = pd.read_csv(CACHE).set_index('node_id')
    if set(cache.index) != node_set:
        raise ValueError('Cached 3D node IDs do not match; inspect or remove cache')
    xyz = cache.loc[nodes, ['x', 'y', 'z']].to_numpy(float)
    print('Loaded saved 3D Spring coordinates (no recomputation).', flush=True)
else:
    # Normalize previous 2D positions, preserving relative locations and structural shape.
    xy = xy_df.loc[nodes, ['x', 'y']].to_numpy(float).copy()
    xy -= xy.mean(axis=0)
    xy /= max(np.max(np.linalg.norm(xy, axis=1)), 1e-12)
    communities = metrics.loc[nodes, 'community_id'].astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    # Nodes in a community start at a similar depth, with gentle individual noise.
    unique = sorted(set(communities))
    community_z = {c: rng.uniform(-1, 1) for c in unique}
    z = Z_STRENGTH * np.array([community_z[c] for c in communities])
    z += rng.normal(0, Z_STRENGTH * 0.18, len(nodes))
    init = {node: np.array([xy[i, 0], xy[i, 1], z[i]], dtype=float)
            for i, node in enumerate(nodes)}
    print(f'Optimizing true 3D Spring layout ({ITERATIONS} iterations)...', flush=True)
    start = time.perf_counter()
    # Every node's x, y and z can move; network edges govern the forces.
    result = nx.spring_layout(G, dim=3, pos=init, iterations=ITERATIONS, seed=SEED)
    xyz = np.array([result[n] for n in nodes], dtype=float)
    print(f'3D Spring completed in {time.perf_counter() - start:.1f}s.', flush=True)
    pd.DataFrame({'node_id': nodes, 'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2]}).to_csv(CACHE, index=False)
    print(f'Saved coordinates: {CACHE}', flush=True)

if not np.isfinite(xyz).all():
    raise ValueError('3D coordinates contain non-finite values')
# Normalize for the visualization only; CSV retains the computed coordinates.
xyz = xyz - xyz.mean(axis=0)
xyz = xyz / max(np.abs(xyz).max(), 1e-12)
index = {n: i for i, n in enumerate(nodes)}
communities = metrics.loc[nodes, 'community_id'].astype(int).to_numpy()
degree = metrics.loc[nodes, 'degree'].astype(float).to_numpy()
between = metrics.loc[nodes, 'betweenness_approx'].astype(float).to_numpy()
unique = sorted(set(communities))
rank = {c: i for i, c in enumerate(unique)}

def color_for(c):
    hue = (0.52 + rank[c] * 0.61803398875) % 1
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return f'rgb({int(255*r)},{int(255*g)},{int(255*b)})'

xe, ye, ze = [], [], []
for u, v in G.edges():
    a, b = xyz[index[u]], xyz[index[v]]
    xe.extend([a[0], b[0], None])
    ye.extend([a[1], b[1], None])
    ze.extend([a[2], b[2], None])

edges = go.Scatter3d(
    x=xe, y=ye, z=ze, mode='lines',
    line=dict(color='rgba(140,177,226,0.32)', width=1),
    hoverinfo='skip', showlegend=False, name='Grid links'
)
points = go.Scatter3d(
    x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers',
    marker=dict(size=np.clip(1.7 + 0.65*np.sqrt(degree), 2, 5.5),
                color=[color_for(c) for c in communities], opacity=0.89),
    customdata=np.column_stack([nodes, communities, degree.astype(int), np.round(between, 5)]),
    hovertemplate='Node %{customdata[0]}<br>Community %{customdata[1]}'
                  '<br>Degree %{customdata[2]}<br>Approx. betweenness %{customdata[3]}<extra></extra>',
    showlegend=False, name='Grid facilities'
)
fig = go.Figure(data=[edges, points])
fig.update_layout(
    title=dict(text='THE NEURAL ANATOMY OF THE GRID — 3D SPRING TEST', x=0.5),
    paper_bgcolor='#07101e', font=dict(color='#e5f1ff'),
    scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
               zaxis=dict(visible=False), bgcolor='#07101e',
               aspectmode='data', camera=dict(eye=dict(x=1.5, y=1.5, z=0.85))),
    margin=dict(l=0, r=0, b=0, t=70),
    annotations=[dict(x=0.5, y=0.02, xref='paper', yref='paper',
                      text=f'{len(nodes):,} nodes  |  {G.number_of_edges():,} edges  |  {len(unique)} communities  |  Drag to rotate',
                      showarrow=False, font=dict(size=12, color='#adc1d8'))]
)
fig.write_html(HTML, include_plotlyjs=True, full_html=True)
print(f'Interactive HTML saved: {HTML.resolve()}', flush=True)
print('Note: z is topology-based visualization depth, not physical elevation.', flush=True)
