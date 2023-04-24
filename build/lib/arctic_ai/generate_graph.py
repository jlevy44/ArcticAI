"""
Graph Data 
==========
Functions for graph dataset generation.
"""
import os, torch, numpy as np, pandas as pd
import pickle
import scipy.sparse as sps
from torch_geometric.utils import subgraph, add_remaining_self_loops
from torch_cluster import radius_graph
from collections import Counter
from torch_geometric.data import Data

def create_graph_data(basename="163_A1a",
                      analysis_type="tumor",
                      radius=256,
                      min_component_size=600,
                      no_component_break=False,
                      dirname="."):
    """
    Creates graph data for use in the GNN model for a given tissue slide.

    Parameters
    ----------
    basename : str
        The basename of the tissue slide to create graph data for.
    analysis_type : str
        The type of analysis to perform. Can be "tumor" or "macro".
    radius : int
        The radius to use when creating the graph.
    min_component_size : int
        The minimum size a connected component must be to be included in the graph data.
    no_component_break : bool
        Whether to include all connected components in the graph data, or just the largest one.
    dirname : str
        The directory to save the graph data in.

    Returns
    -------
    None
    """

    os.makedirs(os.path.join(dirname,"graph_datasets"),exist_ok=True)

    embeddings=torch.load(os.path.join(dirname,f"cnn_embeddings/{basename}_{analysis_type}_map.pkl"))
    xy=torch.tensor(embeddings['patch_info'][['x','y']].values).float()
    if torch.cuda.is_available():
        xy=xy.cuda()
    X=torch.tensor(embeddings['embeddings'])
    G=radius_graph(xy, r=radius*np.sqrt(2), batch=None, loop=True)
    G=G.detach().cpu()
    G=add_remaining_self_loops(G)[0]
    xy=xy.detach().cpu()
    datasets=[]
    edges=G.detach().cpu().numpy().astype(int)
    n_components,components=list(sps.csgraph.connected_components(sps.coo_matrix((np.ones_like(edges[0]),(edges[0],edges[1])))))
    comp_count=Counter(components)
    components=torch.LongTensor(components)
    if not no_component_break:
        for i in range(n_components):
            if comp_count[i]>=min_component_size:
                G_new=subgraph(components==i,G,relabel_nodes=True)[0]
                xy_new=xy[components==i]
                X_new=X[components==i]
                np.random.seed(42)
                idx=np.arange(X_new.shape[0])
                idx2=np.arange(X_new.shape[0])
                np.random.shuffle(idx)
                train_idx,val_idx,test_idx=torch.tensor(np.isin(idx2,idx[:int(0.8*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.8*len(idx)):int(0.9*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.9*len(idx)):]))
                dataset=Data(x=X_new, edge_index=G_new, y_new=torch.ones(len(X_new)), edge_attr=None, pos=xy_new)
                dataset.train_mask=train_idx
                dataset.val_mask=val_idx
                dataset.test_mask=test_idx
                dataset.id=basename
                dataset.component=i
                datasets.append(dataset)
    else:
        dataset=Data(x=X, edge_index=G, y_new=torch.ones(len(X)), edge_attr=None, pos=xy)
        np.random.seed(42)
        idx=np.arange(X.shape[0])
        idx2=np.arange(X.shape[0])
        np.random.shuffle(idx)
        train_idx,val_idx,test_idx=torch.tensor(np.isin(idx2,idx[:int(0.8*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.8*len(idx)):int(0.9*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.9*len(idx)):]))
        dataset.train_mask=train_idx
        dataset.val_mask=val_idx
        dataset.test_mask=test_idx
        dataset.id=basename
        dataset.component=0
        datasets.append(dataset)

    pickle.dump(datasets,open(os.path.join(dirname,'graph_datasets',f"{basename}_{analysis_type}_map.pkl"),'wb'))
