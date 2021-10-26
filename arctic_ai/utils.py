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

def return_osd_template():
    raise NotImplementedError
    return """<html>
<head>
    <link rel="stylesheet" href="style.css">
    <title id="title">basename</title>
    <meta http-equiv="Cache-control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
</head>
<body>
    <div class="float-container">
        <div>
            <span>Modes</span>
            <button id="curtainButton">Curtain</button>
            <button id="syncButton">Sync</button>
        </div>
        Image 1: &nbsp <select id="tileSourceSelect1"></select> <br>
        Image 2: <select id="tileSourceSelect2"></select> <br>
        <div class="float-child float-left">
            <span class="image-title" id="leftImageText"></span> <br>
        </div>
        <div class="float-child float-right">
            <span class="image-title" id="rightImageText"></span> <br>
        </div>
    </div>
    <div class="openseadragon" id="viewer"></div>
    <script src="/openseadragon/openseadragon.min.js"></script>
    <script src="/openseadragon/openseadragon-curtain-sync.min.js"></script>
    <script type="text/javascript">

        var imagePrefix = 0;

        var allTileSources = ["REPLACE"];
        // for (var i = 0; i <= 2; i += 1) {
        //    allTileSources.push("images/140_A1c.img."+i+".dzi");
        //    allTileSources.push("images/140_A1c.prob_map."+i+".dzi");
        //}
        console.log(allTileSources);
        document.getElementById("title").innerHTML = "BASENAME";

        var viewerImages = [];
        for (var i = 0; i < allTileSources.length; i++) {
            viewerImages.push({
                key: allTileSources[i],
                tileSource: allTileSources[i],
            });
        }
        // create viewers
        var viewer = new CurtainSyncViewer({
            mode: 'sync',
            container: document.getElementById("viewer"),
            images: viewerImages,
        });

        var curtainButton = document.getElementById("curtainButton");
        var syncButton = document.getElementById("syncButton");
        curtainButton.onclick = function() {
            viewer.setMode("curtain");
        };
        syncButton.onclick = function() {
            viewer.setMode("sync");
        };



        // setup drop-down menus
        var tileSourceSelect1 = document.getElementById("tileSourceSelect1");
        var tileSourceSelect2 = document.getElementById("tileSourceSelect2");
        for (var i = 0; i < allTileSources.length; i++) {
            tileSourceSelect1.appendChild(new Option(allTileSources[i], allTileSources[i]));
            tileSourceSelect2.appendChild(new Option(allTileSources[i], allTileSources[i]));
        }

        // default images shown
        var image1 = allTileSources[0];
        var image2 = allTileSources[1];
        viewer.setImageShown(image1, true)
        viewer.setImageShown(image2, true)
        tileSourceSelect1.value = image1;
        tileSourceSelect2.value = image2;
        var leftImageText = document.getElementById("leftImageText");
        var rightImageText = document.getElementById("rightImageText");
        leftImageText.innerHTML = image1;
        rightImageText.innerHTML = image2;

        function setImageOrderText() {
            if (allTileSources.indexOf(image1) < allTileSources.indexOf(image2)) {
                leftImageText.innerHTML = image1;
                rightImageText.innerHTML = image2;
            } else {
                leftImageText.innerHTML = image2;
                rightImageText.innerHTML = image1;
            }
        }

        // setup dropdown menu actions
        tileSourceSelect1.onchange = function(a) {
            console.log("Left image:", this.value);
            if (this.value != image1 && this.value != image2) {
                var curZoom = viewer.getZoom();

                viewer.setImageShown(image1, false);
                image1 = this.value;
                viewer.setImageShown(image1, true);
                setTimeout(function() {
                    viewer.setZoom(curZoom);
                }, 100);

                setImageOrderText();
            } else {
                console.log("EXCEPTION: Selected image is already displayed.");
            }
        }
        tileSourceSelect2.onchange = function(a) {
            console.log("Image:", this.value);
            if (this.value != image1 && this.value != image2) {
                var curZoom = viewer.getZoom();

                viewer.setImageShown(image2, false);
                image2 = this.value;
                viewer.setImageShown(image2, true);
                setTimeout(function() {
                    viewer.setZoom(curZoom);
                }, 100);

                setImageOrderText();
            } else {
                console.log("EXCEPTION: Selected image is already displayed.");
            }
        }

    </script>
</body>
</html>
​"""
