import sys, os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pickle
import os,glob, pandas as pd
from sklearn.metrics import f1_score
import copy
from collections import Counter
import torch
import torch.nn.functional as F
import traceback
try:
    from torch import nn
    from torch_geometric.nn import GCNConv, GATConv, DeepGraphInfomax, SAGEConv
    from torch_geometric.nn import DenseGraphConv
    from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse
    from torch_geometric.nn import GINEConv
    from torch_geometric.utils import dropout_adj
    from torch_geometric.nn import APPNP
    from torch_cluster import knn_graph
    from torch_geometric.data import Data 
    from torch_geometric.utils import train_test_split_edges
    from torch_geometric.utils.convert import to_networkx
    from torch_geometric.data import InMemoryDataset,DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
except:
    print(traceback.format_exc())


def get_graph_dataset(basename, embedding_dir='gcn/nodes', k=8, radius=0):
    df=pickle.load(open(os.path.join(embedding_dir,f"{basename}.pkl"), "rb"))
    xy=torch.tensor(df[['X','Y']].values).cuda()
    
    
    X=torch.tensor(np.stack([x for x in df.Embedding.values]), dtype=torch.float32)
    y=torch.tensor(df.Label.values)
    # needs to be modified... build a graph for each individual patch
    if not radius:
        G=knn_graph(xy,k=k)
    else:
        G=radius_graph(xy, r=radius*np.sqrt(2), batch=None, loop=True)
        
    G=G.detach().cpu()
    G=torch_geometric.utils.add_remaining_self_loops(G)[0]
    xy=xy.detach().cpu()
    datasets=[]
    edges=G.detach().cpu().numpy().astype(int)
    n_components,components=list(sps.csgraph.connected_components(sps.coo_matrix((np.ones_like(edges[0]),(edges[0],edges[1])))))
    components=torch.LongTensor(components)
    for i in range(n_components):
        G_new=subgraph(components==i,G,relabel_nodes=True)[0]
        xy_new=xy[components==i]
        X_new=X[components==i]
        y_new=y[components==i]
        np.random.seed(42)
        idx=np.arange(X_new.shape[0])
        idx2=np.arange(X_new.shape[0])
        np.random.shuffle(idx)
        train_idx,val_idx,test_idx=torch.tensor(np.isin(idx2,idx[:int(0.8*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.8*len(idx)):int(0.9*len(idx))])),torch.tensor(np.isin(idx2,idx[int(0.9*len(idx)):]))
        dataset=Data(x=X_new, edge_index=G_new, edge_attr=None, y=y_new, pos=xy_new)
        dataset.train_mask=train_idx
        dataset.val_mask=val_idx
        dataset.test_mask=test_idx
        datasets.append(dataset)
    return datasets

EPS = 1e-15

class GCNNet(torch.nn.Module):
    def __init__(self, inp_dim, out_dim, hidden_topology=[32,64,128,128], p=0.5, p2=0.1, drop_each=True):
        super(GCNNet, self).__init__()
        self.out_dim=out_dim
        self.convs = nn.ModuleList([GATConv(inp_dim, hidden_topology[0])]+[GATConv(hidden_topology[i],hidden_topology[i+1]) for i in range(len(hidden_topology[:-1]))])
        self.drop_edge = lambda edge_index: dropout_adj(edge_index,p=p2)[0]
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(hidden_topology[-1], out_dim)
        self.drop_each=drop_each

    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.convs:
            if self.drop_each and self.training: edge_index=self.drop_edge(edge_index)
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
        if self.training:
            x = self.dropout(x)
        x = self.fc(x)
        return x
    
class GCNFeatures(torch.nn.Module):
    def __init__(self, gcn, bayes=False):
        super(GCNFeatures, self).__init__()
        self.gcn=gcn
        self.drop_each=bayes
    
    def forward(self, x, edge_index, edge_attr=None):
        for conv in self.gcn.convs:
            if self.drop_each: edge_index=self.gcn.drop_edge(edge_index)
            x = F.relu(conv(x, edge_index, edge_attr))
        if self.drop_each:
            x = self.gcn.dropout(x)
        y = F.softmax(self.gcn.fc(x))
        return x,y

def validate_model(graph_data="graph_data.pkl",
                use_weights=False,
                use_model=None,
                n_batches_backward=1,
                f1_metric='weighted',
                n_epochs=1500,
                out_dir='gcn/models',
                lr=1e-2,
                eta_min=1e-4,
                T_max=20,
                wd=0,
                hidden_topology=[32,64,128,128],
                p=0.5,
                p2=0.3,
                burnin=400,
                warmup=100,
                gpu_id=0,
                batch_size=1
                ):
    print(gpu_id); torch.cuda.set_device(gpu_id)
    datasets=pickle.load(open(graph_data, "rb"))
    train_dataset = [dataset for key in datasets for dataset in datasets[key]['train']]
    val_dataset = [dataset for key in datasets for dataset in datasets[key]['val']]
    
   
    y_train=np.hstack([graph.y.numpy() for graph in train_dataset])
    if use_weights: 
        weights=compute_class_weight('balanced',classes=np.unique(y_train),y=y_train)
    else: 
        weights=None
    print("number of classes", len(np.unique(y_train)))
    # load model
    model=GCNNet(train_dataset[0].x.shape[1],len(np.unique(y_train)),hidden_topology=hidden_topology,p=p,p2=p2)
    if use_model:
        model.load_state_dict(torch.load(use_model))
    model=model.cuda()
    
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights).float() if use_weights else None)
    criterion=criterion.cuda()
    
    Y_all = {}
    Y_Pred_all = {}
    pos_all = {}
    Y,Y_Pred=[],[]
    pos = []
    wsi = []    
    for key in datasets:
        train_dataset = datasets[key]['train']
        val_dataset = datasets[key]['val']
        dataloaders={}

        dataloaders['train']=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
        dataloaders['warmup']=DataLoader(train_dataset,shuffle=False)
        train_loader=dataloaders['warmup']
    
        

        for i,data in enumerate(dataloaders['train']):
            model.train(False)
            x=data.x.cuda()
            edge_index=data.edge_index.cuda()
            y=data.y.cuda()
            y_out=model(x,edge_index)
            #print(y_out, y)
            loss = criterion(y_out, y) 
            y_prob=F.softmax(y_out, dim=1).detach().cpu().numpy()
            y_pred=y_prob.argmax(1).flatten()
            y_true=y.detach().cpu().numpy().flatten()
            Y_Pred.append(y_pred)
            Y.append(y_true)
            pos.append(data.pos)
            for i in range(len(y_pred)):
                wsi.append(key)
            del x, edge_index, loss, y_out
        if (len(val_dataset) > 0):
            dataloaders['val']=DataLoader(val_dataset,shuffle=True)
            for i,data in enumerate(dataloaders['val']):
                model.train(False)
                x=data.x.cuda()
                edge_index=data.edge_index.cuda()
                y=data.y.cuda()
                y_out=model(x,edge_index)
                loss = criterion(y_out, y) 
                y_prob=F.softmax(y_out, dim=1).detach().cpu().numpy()
                y_pred=y_prob.argmax(1).flatten()
                y_true=y.detach().cpu().numpy().flatten()
                Y_Pred.append(y_pred)
                Y.append(y_true)
                pos.append(data.pos)
                for i in range(len(y_pred)):
                    wsi.append(key)
                del x, edge_index, loss, y_out
        
    Y = np.hstack(Y)
    Y_Pred = np.hstack(Y_Pred)
    pos = np.vstack(pos)
        
    return Y, Y_Pred, pos, wsi