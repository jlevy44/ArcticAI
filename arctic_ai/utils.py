def tif2npy(in_file,out_dir,overwrite=False):
    import os, numpy as np
    import tifffile
    basename,ext=os.path.splitext(os.path.basename(in_file))
    out_file=os.path.join(out_dir,f"{basename}.npy")
    if not os.path.exists(out_file) or overwrite:
        np.save(out_file,tifffile.imread(in_file))

def display_results(out_graphs,res_,predict=False,custom_colors=[],s=1,img=None,alpha=None,scatter=True,scale=8,width_scale=20,node_scale=90,preds=None):
    import matplotlib
    import networkx as nx
    matplotlib.rcParams['figure.dpi']=300
    matplotlib.rcParams['axes.grid'] = False
    import matplotlib.pyplot as plt
    import cv2, numpy as np, pandas as pd
    # from arctic_ai.dgm.dgm.plotting import *
    import copy

    f = plt.figure(figsize=(15,15))
    ax = f.add_subplot(1, 1, 1)
    binary=False
    if not isinstance(img,type(None)): plt.imshow(np.transpose(img,(1,0,2)))

    for out_graph,res,pred in zip(out_graphs,res_,preds):
        xy=pred["xy"]
        y_orig=pred["y"]
        y=copy.deepcopy(y_orig)
        graph=out_graph
        node_color=res['mnode_to_color']; node_size=res['node_sizes']; edge_weight=res['edge_weight']
        if custom_colors: node_color=custom_colors
        node_list=res['node_list']; name='wsi'
        cmap = cm.coolwarm
        cmap = cm.get_cmap(cmap, 100)
        plt.set_cmap(cmap)
        edges = graph.edges()
        weights = np.array([edge_weight[(min(u, v), max(u, v))] for u, v in edges], dtype=np.float32)
        width = weights * width_scale
        node_size = np.sqrt(node_size) * node_scale
        c=y.flatten()
        pos = {}
        for node in graph.nodes():
            if len(res['mnode_to_nodes'][node])-1:
                pos[node]=np.array([xy[i] for i in res['mnode_to_nodes'][node]]).mean(0)/scale
            else:
                pos[node]=xy[list(res['mnode_to_nodes'][node])[0]]/scale
        if scatter: plt.scatter(xy[:,0]/scale,xy[:,1]/scale,c=c,alpha=alpha,s=s)
        nx.draw(graph, pos=pos, node_color=node_color, width=width, node_size=node_size,
                node_list=node_list, ax=ax, cmap=cmap)
    plt.axis('off')
    plt.grid(b=None)
    return None

def plot_results(basename="163_A1c",
                 compression=4,
                 k='macro'):
    import cv2, numpy as np, pandas as pd
    img=np.load(f"inputs/{basename}.npy")
    im=cv2.resize(img,None,fx=1/compression,fy=1/compression)
    mapper_graphs=pd.read_pickle(f"mapper_graphs/{basename}.pkl")
    for i in range(len(mapper_graphs[k])):
        mapper_graphs[k][i]['graph']['y']=mapper_graphs[k][i]['graph']['y_pred'].argmax(1)
    out_graphs,res_,preds=[mapper_graphs[k][i]['out_res'][0] for i in range(len(mapper_graphs[k]))],[mapper_graphs[k][i]['out_res'][1] for i in range(len(mapper_graphs[k]))],[mapper_graphs[k][i]['graph'] for i in range(len(mapper_graphs[k]))]
    display_results(out_graphs,res_,alpha=0.2,s=20,img=im,preds=preds,scale=compression,node_scale=30)

def try_except(fn,*args,**kwargs):
    try:
        fn(*args,**kwargs)
    except:
        pass

def return_osd_template():
    import os, subprocess
    try_except(os.makedirs,"dzi_files",exist_ok=True)
    for f in ['osd_scripts','dzi_files/openseadragon','dzi_files/style.css','./osd_template.html']:
        try_except(os.system,f"rm -rf ${f}")
    try_except(os.system,"git clone https://github.com/jlevy44/osd_scripts")
    try_except(os.link,"osd_scripts/osd_template.html","./osd_template.html")
    try_except(os.link,"osd_scripts/style.css","dzi_files/style.css")
    try_except(os.system,"cp -al osd_scripts/openseadragon dzi_files/")
