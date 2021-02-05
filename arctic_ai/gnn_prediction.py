import os, torch, pickle, numpy as np, pandas as pd, torch.nn as nn
from torch_geometric.data import DataLoader as TG_DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj, dense_to_sparse, dropout_adj, to_networkx
from torch_geometric.nn import GATConv
import torch.nn.functional as F

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
            x = F.relu(conv(x, edge_index, edge_attr))
        if self.training:
            x = self.dropout(x)
        x = self.fc(x)
        return x

class GCNFeatures(torch.nn.Module):
    def __init__(self, gcn, bayes=False, p=0.05, p2=0.1):
        super(GCNFeatures, self).__init__()
        self.gcn=gcn
        self.drop_each=bayes
        self.gcn.drop_edge = lambda edge_index: dropout_adj(edge_index,p=p2)[0]
        self.gcn.dropout = nn.Dropout(p)

    def forward(self, x, edge_index, edge_attr=None):
        for i,conv in enumerate(self.gcn.convs):
            if self.drop_each: edge_index=self.gcn.drop_edge(edge_index)
            x = conv(x, edge_index, edge_attr)
            if i+1<len(self.gcn.convs):
                x=F.relu(x)
        if self.drop_each:
            x = self.gcn.dropout(x)
        y = self.gcn.fc(F.relu(x))#F.softmax()
        return x,y

def predict(basename="163_A1a",
            analysis_type="tumor",
            gpu_id=0):
    hidden_topology=dict(tumor=[32,64,64],macro=[32,64,64])#[32]*3
    num_classes=dict(macro=4,tumor=3)
    if gpu_id>=0: torch.cuda.set_device(gpu_id)
    dataset=pickle.load(open(os.path.join('graph_datasets',f"{basename}_{analysis_type}_map.pkl"),'rb'))
    model=GCNNet(dataset[0].x.shape[1],num_classes[analysis_type],hidden_topology=hidden_topology[analysis_type],p=0.,p2=0.)
    model=model.cuda()
    model.load_state_dict(torch.load(os.path.join("models",f"{analysis_type}_map_gnn.pth"),map_location=f"cuda:{gpu_id}" if gpu_id>=0 else "cpu"))
    dataloader=TG_DataLoader(dataset,shuffle=False,batch_size=1)
    model.eval()
    feature_extractor=GCNFeatures(model,bayes=False).cuda()
    graphs=[]
    for i,data in enumerate(dataloader):
        with torch.no_grad():
            graph = to_networkx(data).to_undirected()
            model.train(False)
            x=data.x.cuda()
            xy=data.pos.numpy()
            edge_index=data.edge_index.cuda()
            preds=feature_extractor(x,edge_index)
            z,y_pred=preds[0].detach().cpu().numpy(),preds[1].detach().cpu().numpy()
            graphs.append(dict(G=graph,xy=xy,z=z,y_pred=y_pred,slide=data.id,component=data.component))
    torch.save(graphs,os.path.join("gnn_results",f"{basename}_{analysis_type}_map.pkl"))
