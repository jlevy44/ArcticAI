from torch_cluster import nearest
import sys, os, torch, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from dgm.dgm import DGM
from umap import UMAP
import pickle

classes_=['dermis', 'epidermis', 'hole', 'subcutaneous tissue']

def relabel_tumor(graph_tumor,graph_macro):
    le=LabelEncoder().fit(classes_)
    re_idx=nearest(torch.tensor(graph_tumor['xy']), torch.tensor(graph_macro['xy'])).numpy()
    unassigned=(graph_tumor['xy']-graph_macro['xy'][re_idx]).sum()!=0
    macro_pred=graph_macro['y_pred'].argmax(1)
    tumor_pred=graph_tumor['y_pred'].argmax(1)
    benign=tumor_pred!=2#0 <- former model
    tumor_pred=tumor_pred.astype('str')
    tumor_pred[benign]=le.inverse_transform(macro_pred[re_idx][benign])
    tumor_pred[~benign]='tumor'
    tumor_pred[unassigned]='unassigned'
    graph_tumor['annotation']=tumor_pred
    return graph_tumor

def construct_mapper(graph):
    z=UMAP(n_components=2,random_state=42).fit_transform(graph['z'])
    return dict(out_res=DGM(num_intervals=2,overlap=0.01,min_component_size=100,eps=0.1, sdgm=True).fit_transform(graph['G'], z),graph=graph)

def get_interaction(out_graph,y_orig,res,lb=None,plot=False,le=None):
    if not isinstance(lb,type(None)):
        y_orig=lb.transform(y_orig)
    node_makeup={}# only if predict
    for node in out_graph.nodes():
        nodes=res['mnode_to_nodes'][node]
        node_makeup[node]=y_orig[np.array(list(nodes))].mean(0)
    edges = out_graph.edges()
    edge_weight=res['edge_weight']
    weights = np.array([edge_weight[(min(u, v), max(u, v))] for u, v in edges], dtype=np.float32)
    edgelist=list(edges)
    A=np.zeros((len(lb.classes_),len(lb.classes_)))
    for i in range(len(edgelist)):
        send=node_makeup[edgelist[i][0]]
        receive=node_makeup[edgelist[i][1]]
        a=np.outer(send,receive)
        a=(a+a.T)/2.*weights[i]
        A+=a
    invasion_mat=pd.DataFrame(A,columns=le.inverse_transform(np.arange(len(lb.classes_))),index=le.inverse_transform(np.arange(len(lb.classes_))))
    return invasion_mat

def calc_hole_vals(dgm_result,weights={'dermis':1,'epidermis':1,'subcutaneous tissue':1}):
    y_pred=dgm_result['graph']['y_pred'].argmax(1)
    out_graph,res=dgm_result['out_res']
    le=LabelEncoder().fit(classes_)
    area_hole=(le.inverse_transform(y_pred)=='hole').mean()
    hole_share=get_interaction(out_graph,y_pred,res,lb=LabelBinarizer().fit(np.arange(len(le.classes_))),le=le)['hole']
    hole_share=hole_share.loc[hole_share.index!='hole']
    hole_share=pd.DataFrame(hole_share).reset_index()
    hole_share2=hole_share.set_index('index')
    hole_share2['weight']=pd.Series(weights)
    hole_share2['importance']=hole_share2['weight']*hole_share2['hole']
    return hole_share2

def calc_tumor_vals(dgm_result,weights={'dermis':1,'epidermis':1,'subcutaneous tissue':1,'hole':1}):
    out_graph,res=dgm_result['out_res']
    le=LabelEncoder().fit(dgm_result['graph']['annotation'])
    y=le.transform(dgm_result['graph']['annotation'])
    tumor_share=get_interaction(out_graph,y,res,lb=LabelBinarizer().fit(np.arange(len(le.classes_))),le=le)['tumor']
    tumor_share=tumor_share.loc[~tumor_share.index.isin(['tumor','unassigned'])]
    tumor_share=pd.DataFrame(tumor_share).reset_index()
    tumor_share2=tumor_share.set_index('index')
    tumor_share2['weight']=pd.Series(weights)
    tumor_share2['importance']=tumor_share2['weight']*tumor_share2['tumor']
    return tumor_share2

def generate_quality_scores(basename):
    graphs={k:torch.load(os.path.join("gnn_results",f"{basename}_{k}_map.pkl")) for k in ['tumor','macro']}
    graphs['tumor']=[relabel_tumor(graph_tumor,graph_macro) for graph_tumor,graph_macro in zip(graphs['tumor'],graphs['macro'])]

    mapper_graphs=dict()
    for k in ['tumor','macro']:
        mapper_graphs[k]=[construct_mapper(graph) for graph in graphs[k]]

    scoring_fn=dict(tumor=calc_tumor_vals,macro=calc_hole_vals)
    quality_score=dict()

    for k in mapper_graphs:
        quality_score[k]=pd.concat([scoring_fn[k](dgm_result)['importance'] for dgm_result in mapper_graphs[k]],axis=1)
        quality_score[k].columns=[f'section_{i}' for i in range(1,len(quality_score[k].columns)+1)]

    pickle.dump(mapper_graphs,open(f'mapper_graphs/{basename}.pkl','wb'))
    pickle.dump(quality_score,open(f'quality_scores/{basename}.pkl','wb'))
