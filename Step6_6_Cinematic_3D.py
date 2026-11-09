"""SYSE580 Step 6.6: cinematic render of the genuine Western US grid.

Place beside network.gml/network.gml, Step5_Node_Centrality.csv, and
EITHER Step6_4_Positions_3D_spring.csv OR Step6_5_C_Side.html.
This script does NOT recompute topology, community detection or layouts.
Outputs an interactive HTML and a cinematic PNG preview.
"""
from pathlib import Path
import colorsys, json
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
GML = ROOT / 'network.gml' / 'network.gml'
if not GML.is_file():
    GML = ROOT / 'network.gml'
METRICS = ROOT / 'Step5_Node_Centrality.csv'
POS = ROOT / 'Step6_4_Positions_3D_spring.csv'
HTML_BACKUP = ROOT / 'Step6_5_C_Side.html'
OUT = ROOT / 'Step6_6_Output'
OUT.mkdir(exist_ok=True)

G = nx.parse_gml(GML.read_text(encoding='utf-8').replace('_pos','pos'), label='id')
node_ids = list(G.nodes())
meta = pd.read_csv(METRICS).set_index('node_id').loc[node_ids]

if POS.is_file():
    df = pd.read_csv(POS).set_index('node_id').loc[node_ids]
    coords = df[['x','y','z']].to_numpy(float)
    # Step 6.5 applied this centering and normalization.
    coords -= coords.mean(axis=0)
    coords /= max(float(np.ptp(coords,axis=0).max()), 1e-12)
    print('Using cached 3D Spring positions')
elif HTML_BACKUP.is_file():
    # Fallback: restore the already-rendered positions without recalculating.
    html = HTML_BACKUP.read_text(encoding='utf-8')
    start = html.rfind('Plotly.newPlot(')
    start = html.index('[{', start)
    traces, _ = json.JSONDecoder().raw_decode(html[start:])
    nodes_trace = next(t for t in traces if t.get('name') == 'Grid nodes')
    node_map = {int(c[0]):(x,y,z) for c,x,y,z in zip(
        nodes_trace['customdata'],nodes_trace['x'],nodes_trace['y'],nodes_trace['z'])}
    coords = np.array([node_map[n] for n in node_ids], float)
    print('Restored cached 3D Spring positions from Step 6.5 HTML')
else:
    raise FileNotFoundError('Need Step6_4_Positions_3D_spring.csv or Step6_5_C_Side.html')
assert np.isfinite(coords).all() and len(coords)==4941
idx={node:i for i,node in enumerate(node_ids)}
community=meta.community_id.to_numpy(int)
degree=meta.degree.to_numpy(float)
between=meta.betweenness_approx.to_numpy(float)
unique=sorted(set(community)); rank={c:i for i,c in enumerate(unique)}

def color(c):
    h=(.52+rank[c]*.61803398875)%1
    return colorsys.hsv_to_rgb(h,.66,.96)

rgb=np.array([color(c) for c in community]); edge_ij=np.array([(idx[u],idx[v]) for u,v in G.edges()])
# C — Side camera, kept identical to Step 6.5.
eye=np.array([2.25,.30,.48]); eye=eye/np.linalg.norm(eye)
forward=-eye
right=np.cross(forward, [0,0,1]);right/=np.linalg.norm(right)
up=np.cross(right,forward)
# Orthographic-style projection for high-resolution poster: easy to inspect.
xp=coords@right; yp=coords@up; depth=coords@eye
span=max(np.ptp(xp),np.ptp(yp))
xp=(xp-np.median(xp))/span; yp=(yp-np.median(yp))/span
normdepth=(depth-depth.min())/(np.ptp(depth)+1e-12)
# 4K 16:9 export
W,H=3840,2160
fig=plt.figure(figsize=(W/160,H/160),dpi=160,facecolor='#070e1d')
ax=fig.add_axes([.04,.14,.92,.74]);ax.set_facecolor('#070e1d')
ax.set_aspect('equal');ax.axis('off')
# Edges: genuine segments only, no synthetic paths.
segments=np.stack((np.stack((xp[edge_ij[:,0]],yp[edge_ij[:,0]]),axis=1),
                   np.stack((xp[edge_ij[:,1]],yp[edge_ij[:,1]]),axis=1)),axis=1)
edgecolor=(rgb[edge_ij[:,0]]+rgb[edge_ij[:,1]])/2
edge_rgba=np.column_stack([edgecolor,np.full(len(edgecolor),.26)])
ax.add_collection(LineCollection(segments, colors=edge_rgba, linewidths=.43, zorder=1))
# Draw faint background stars? No: avoids fabricating extra nodes.
# Gaussian glow is computed from SAME real nodes and edges.
ax.scatter(xp,yp,s=1.5+2.3*np.sqrt(degree),c=rgb,alpha=.30,linewidths=0,zorder=3)
ax.scatter(xp,yp,s=.6+.8*np.sqrt(degree),c=rgb,alpha=.94,linewidths=0,zorder=4)
key=np.where(between>=np.quantile(between,.99))[0]
ax.scatter(xp[key],yp[key],s=7+12*np.sqrt(degree[key]),c='#e5f8ff',alpha=.90,linewidths=0,zorder=6)
ax.set_xlim(xp.min()-.055,xp.max()+.055)
ax.set_ylim(yp.min()-.07,yp.max()+.07)
fig.text(.5,.945,'THE NEURAL ANATOMY OF THE GRID',color='#ecf6ff',ha='center',fontsize=24,weight='bold',family='DejaVu Sans')
fig.text(.5,.909,'WESTERN US POWER GRID    /    THREE-DIMENSIONAL TOPOLOGY',color='#9ab4d1',ha='center',fontsize=10)
fig.text(.5,.067,'4,941 NODES    /    6,594 LINKS    /    42 COMMUNITIES    /    3D SPRING LAYOUT',color='#b7c9de',ha='center',fontsize=10)
fig.text(.5,.044,'COLOR: LOUVAIN COMMUNITY     SIZE: DEGREE     HIGHLIGHT: APPROXIMATE BETWEENNESS',color='#738ba7',ha='center',fontsize=7.5)
fig.text(.5,.022,'ARTISTIC TOPOLOGY RENDERING — NOT GEOGRAPHIC POSITION OR ELECTRICITY FLOW',color='#607892',ha='center',fontsize=7)
raw=OUT/'Step6_6_Cinematic_3D_base.png'
fig.savefig(raw,facecolor='#070e1d',dpi=160);plt.close(fig)
# Small, physically non-meaningful glow post-process: no topology changes.
from PIL import Image
base=np.asarray(Image.open(raw).convert('RGB'),dtype=np.float32)/255
background=np.array([7,14,29],dtype=np.float32)/255
light=np.maximum(base-background,0)
blur=gaussian_filter(light,sigma=(8,8,0))
result=np.clip(base+blur*.95,0,1)
image=Image.fromarray((result*255).astype('uint8'))
final=OUT/'Step6_6_Cinematic_3D_Poster.png'
image.save(final,optimize=True)
print('Saved 4K poster:',final)

# Interactive view (Plotly). Geometric camera unchanged from Step 6.5 C.
ex=[];ey=[];ez=[]
for ia,ib in edge_ij:
    p,q=coords[ia],coords[ib]
    ex.extend([p[0],q[0],None]);ey.extend([p[1],q[1],None]);ez.extend([p[2],q[2],None])
fig3=go.Figure()
fig3.add_trace(go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',line=dict(color='rgba(100,167,230,.34)',width=1.1),hoverinfo='skip',showlegend=False))
fig3.add_trace(go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],mode='markers',
    marker=dict(size=np.clip(.95+.3*np.sqrt(degree),1.2,3),color=['rgb({},{},{})'.format(*[int(255*v) for v in c]) for c in rgb],opacity=.73,line=dict(width=0)),
    customdata=np.column_stack([node_ids,community,degree,np.round(between,5)]),
    hovertemplate='Node %{customdata[0]}<br>Community %{customdata[1]}<br>Degree %{customdata[2]}<br>Approx. betweenness %{customdata[3]}<extra></extra>',showlegend=False))
fig3.add_trace(go.Scatter3d(x=coords[key,0],y=coords[key,1],z=coords[key,2],mode='markers',marker=dict(size=3.7,color='#d2f5ff',opacity=.83,line=dict(width=0)),hoverinfo='skip',showlegend=False))
fig3.update_layout(paper_bgcolor='#070e1d',font=dict(color='#e0efff'),
    scene=dict(bgcolor='#070e1d',xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),aspectmode='data',
    camera=dict(eye=dict(x=2.25,y=.30,z=.48),projection=dict(type='perspective'))),
    title=dict(text='THE NEURAL ANATOMY OF THE GRID  |  CINEMATIC 3D',x=.5),margin=dict(l=0,r=0,b=0,t=75))
out_html=OUT/'Step6_6_Cinematic_3D_Interactive.html'
fig3.write_html(out_html,include_plotlyjs=True)
print('Saved interactive 3D:',out_html)
