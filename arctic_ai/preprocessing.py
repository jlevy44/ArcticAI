"""Preprocess 
==========
Contains functions for preprocessing images."""
import os, tqdm
import numpy as np, pandas as pd
from itertools import product
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from pathpretrain.utils import load_image, generate_tissue_mask
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import SpectralClustering
from shapely.geometry import Point, MultiPoint
import cv2
import dask
from dask.diagnostics import ProgressBar
import alphashape
import warnings
# import pysnooper

def preprocess(basename="163_A1a",
               threshold=0.05,
               patch_size=256,
               ext='.npy',
               secondary_patch_size=0,
               alpha=1024**-1,
               no_break=False,
               df_section_pieces_file='',
               dirname=".",
               image_mask_compression=1.,
               use_section=False
               ):
    """
    Preprocesses image and generates patches for training or testing.

    Parameters
    ----------
    basename: str
        The base filename of the image to be preprocessed.
    threshold: float, optional
        The maximum fraction of a patch that can be blank. Default is 0.05.
    patch_size: int, optional
        The size of the patches to be generated. Default is 256.
    ext: str, optional
        The file extension of the input image. Default is ".npy".
    no_break: bool, optional
        If True, the function will not break large images into multiple smaller ones. Default is False.
    df_section_pieces_file: str, optional
        The filename of the file containing metadata about image patches. Default is "section_pieces.pkl".
    image_mask_compression: float, optional
        The degree of compression applied to the image mask. Default is 8.
    dirname: str, optional
        The directory where input and output files are stored. Default is ".".
    """

    assert secondary_patch_size==0
    write_images=False
    os.makedirs(os.path.join(dirname,"masks"),exist_ok=True)
    os.makedirs(os.path.join(dirname,"patches"),exist_ok=True)
    os.makedirs(os.path.join(dirname,"images"),exist_ok=True)
    os.makedirs(os.path.join(dirname,"metadata"),exist_ok=True)

    image=os.path.join(dirname,"inputs",f"{basename}{ext}")
    basename=os.path.basename(image).replace(ext,'')
    image=load_image(image)#np.load(image)
    img_shape=image.shape[:-1]
    df_section_pieces=None if not (df_section_pieces_file and os.path.exists(df_section_pieces_file)) else pd.read_pickle(df_section_pieces_file).reset_index().drop_duplicates().set_index("index")

    masks=dict()
    masks['tumor_map']=generate_tissue_mask(image,
                             compression=10,
                             otsu=False,
                             threshold=240,
                             connectivity=8,
                             kernel=5,
                             min_object_size=100000,
                             return_convex_hull=False,
                             keep_holes=False,
                             max_hole_size=6000,
                             gray_before_close=True,
                             blur_size=51)

    x_max,y_max=masks['tumor_map'].shape
    if no_break: masks['macro_map']=fill_holes(masks['tumor_map'])

    patch_info=dict()
    patches=dict()
    include_patches=dict()
    patch_info['orig']=pd.DataFrame([[basename,x,y,patch_size,"0"] for x,y in tqdm.tqdm(list(product(range(0,x_max-patch_size,patch_size),range(0,y_max-patch_size,patch_size))))],columns=['ID','x','y','patch_size','annotation'])

    for k in (masks if no_break else ['tumor_map']):
        patch_info[k]=patch_info['orig'].copy()
        include_patches[k]=np.stack([masks[k][x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())]).mean((1,2))>=threshold
        patch_info[k]=patch_info[k][include_patches[k]]

        if no_break:
            patches[k]=np.stack([image[x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())])
            np.save(os.path.join(dirname,"masks",f"{basename}_{k}.npy"),masks[k])
            np.save(os.path.join(dirname,"patches",f"{basename}_{k}.npy"),patches[include_patches])
            patch_info[k].to_pickle(os.path.join(dirname,"patches",f"{basename}_{k}.pkl"))
    if no_break: return None

    if not no_break:
        if df_section_pieces is not None: n_pieces=int(np.prod(df_section_pieces.loc[basename.replace("_ASAP","")]))
        G=radius_neighbors_graph(patch_info['tumor_map'][['x','y']], radius=512*np.sqrt(2))
        patch_info['tumor_map']['piece_ID']=connected_components(G)[1]
        if df_section_pieces is None: n_pieces=int(patch_info['tumor_map']['piece_ID'].max()+1)
        patch_info['tumor_map']['piece_ID']=patch_info['tumor_map']['piece_ID'].max()-patch_info['tumor_map']['piece_ID']
        patch_info['tumor_map']=patch_info['tumor_map'][patch_info['tumor_map']['piece_ID'].isin(patch_info['tumor_map']['piece_ID'].value_counts().index[:n_pieces].values)]
        patch_info['tumor_map']['piece_ID']=patch_info['tumor_map']['piece_ID'].map({v:k for k,v in enumerate(sorted(patch_info['tumor_map']['piece_ID'].unique()))})
        if df_section_pieces is not None:
            assert df_section_pieces.loc[basename.replace("_ASAP","")]['Pieces']<=2
            G=radius_neighbors_graph(patch_info['tumor_map'][['x','y']], radius=4096*np.sqrt(2))
            patch_info['tumor_map']['piece_ID']=patch_info['tumor_map']['piece_ID'].map(dict(zip(patch_info['tumor_map'].groupby("piece_ID")['x'].mean().sort_values(ascending=False).index,range(patch_info['tumor_map']['piece_ID'].max()+1))))
            patch_info['tumor_map']['section_ID']=connected_components(G)[1]
            complete=patch_info['tumor_map'][['section_ID','piece_ID']].groupby("section_ID")['piece_ID'].nunique()==df_section_pieces.loc[basename.replace("_ASAP","")]['Pieces']
            patch_info['tumor_map']['complete']=patch_info['tumor_map']['section_ID'].isin(complete[complete].index)
            while patch_info['tumor_map']['piece_ID'].max()+1<n_pieces:
                split_pieces=patch_info['tumor_map']['piece_ID'].value_counts().index
                for split_piece in split_pieces:
                    if patch_info['tumor_map'].loc[patch_info['tumor_map']['piece_ID']==split_piece,'complete'].sum()==0:
                        patch_info['tumor_map'].loc[patch_info['tumor_map']['piece_ID']==split_piece,'complete']=True # TODO: this can break
                        break
                G=radius_neighbors_graph(patch_info['tumor_map'].query(f"piece_ID=={split_piece}")[['x','y']], radius=patch_size*np.sqrt(2))
                cl=SpectralClustering(n_clusters=2,affinity="precomputed",assign_labels="discretize",eigen_solver="amg",n_components=2).fit_predict(G)
                patch_info['tumor_map'].loc[patch_info['tumor_map']['piece_ID']==split_piece,'piece_ID']=cl+patch_info['tumor_map']['piece_ID'].max()+1
                patch_info['tumor_map']['piece_ID']=patch_info['tumor_map']['piece_ID'].map(dict(zip(patch_info['tumor_map'].groupby("piece_ID")['x'].mean().sort_values(ascending=False).index,range(patch_info['tumor_map']['piece_ID'].max()+1))))
            patch_info['tumor_map']['section_ID']=patch_info['tumor_map']['piece_ID']//df_section_pieces.loc[basename.replace("_ASAP","")]['Pieces']
            assert patch_info['tumor_map']['piece_ID'].max()+1==n_pieces
        else:
            G=radius_neighbors_graph(patch_info['tumor_map'][['x','y']], radius=4096*np.sqrt(2))
            patch_info['tumor_map']['section_ID']=connected_components(G)[1]
            patch_info['tumor_map']['section_ID']=patch_info['tumor_map']['section_ID'].max()-patch_info['tumor_map']['section_ID']
        n_sections=patch_info['tumor_map']['section_ID'].max()+1
        n_pieces_per_section=(patch_info['tumor_map']['piece_ID'].max()+1)/n_sections
        pts=MultiPoint(patch_info['orig'][['x','y']].values)
        patch_info_new=[]
        for ID in patch_info['tumor_map']['piece_ID'].unique():
            tmp_points=patch_info['tumor_map'][['x','y']][patch_info['tumor_map']['piece_ID']==ID].values
            alpha_shape = alphashape.alphashape(tmp_points,alpha=alpha)
            tmp_points=MultiPoint(tmp_points)
            xy=dict()
            xy['macro']=pts.intersection(alpha_shape).difference(alpha_shape.exterior.buffer(256))
            xy['macro_tumor']=xy['macro'].intersection(tmp_points.buffer(64))
            xy['macro_no_tumor']=xy['macro'].difference(tmp_points.buffer(64))
            xy['tumor_no_macro']=tmp_points.difference(xy['macro'].buffer(64))
            for k in list(xy.keys()):
                if isinstance(xy[k],Point) or len(xy[k].geoms)==0:
                    del xy[k]
            del xy['macro']
            xy={k:pd.DataFrame(np.array([(int(p.x),int(p.y)) for p in xy[k]]),columns=['x','y']) for k in xy}
            for k in xy:
                xy[k]['basename']=basename
                xy[k]['section_ID']=ID//n_pieces_per_section
                xy[k]['piece_ID']=ID
                xy[k]['patch_size']=patch_size
                xy[k]['Type']=k
            xy=pd.concat(list(xy.values()),axis=0)
            xy['tumor_map']=xy['Type'].isin(['macro_tumor','tumor_no_macro'])
            xy['macro_map']=xy['Type'].isin(['macro_tumor','macro_no_tumor'])
            patch_info_new.append(xy)

        patch_info=pd.concat(patch_info_new,axis=0)
        for coord in ['x','y']: patch_info[f'{coord}_orig']=patch_info[coord]
        xy_bounds={}
        write_files=[]
        new_basenames=[]

        for ID in patch_info['section_ID' if use_section else 'piece_ID'].unique():
            new_basename=f"{basename}_{ID}"
            new_basenames.append(new_basename)
            include_patches=(patch_info['section_ID' if use_section else 'piece_ID']==ID).values
            patch_info_ID=patch_info[include_patches]
            (xmin,ymin),(xmax,ymax)=patch_info_ID[['x','y']].min(0).values,(patch_info_ID[['x','y']].max(0).values+patch_size)
            im=image[xmin:xmax,ymin:ymax]
            msk=masks['tumor_map'][xmin:xmax,ymin:ymax] # TODO: can break, need target tissue section
            patch_info_ID.loc[:,['x','y']]-=patch_info_ID[['x','y']].min(0)
            patches_ID=np.stack([im[x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info_ID[['x','y']].values.tolist())])
            patch_info_ID.reset_index(drop=True).to_pickle(os.path.join(dirname,"patches",f"{new_basename}.pkl"))
            write_files.append(dask.delayed(np.save)(os.path.join(dirname,"patches",f"{new_basename}.npy"),patches_ID))
            write_files.append(dask.delayed(np.save)(os.path.join(dirname,"masks",f"{new_basename}.npy"),cv2.resize(msk.astype(np.uint8),None,fx=1/image_mask_compression,fy=1/image_mask_compression,interpolation=cv2.INTER_NEAREST)>0 if image_mask_compression>1 else msk))
            if write_images:
                if image_mask_compression>1:
                    write_files.append(dask.delayed(np.save)(os.path.join(dirname,"images",f"{new_basename}.npy"),cv2.resize(im,None,fx=1/image_mask_compression,fy=1/image_mask_compression,interpolation=cv2.INTER_CUBIC)))
                else:
                    write_files.append(dask.delayed(np.save)(os.path.join(dirname,"images",f"{new_basename}.npy"),im))
            xy_bounds[ID]=((xmin,ymin),(xmax,ymax))
        pd.to_pickle(xy_bounds,os.path.join(dirname,"masks",f"{basename}.pkl"))
        with ProgressBar():
            dask.compute(write_files,scheduler='threading')
        pd.to_pickle(new_basenames,os.path.join(dirname,"metadata",f"{basename}.pkl"))
    return None

def preprocess_old(basename="163_A1a",
               threshold=0.05,
               patch_size=256,
               ext='.npy',
               secondary_patch_size=0):

    warnings.warn(
            "Old preprocessing is deprecated",
            DeprecationWarning
        )
    raise RuntimeError
    os.makedirs("masks",exist_ok=True)
    os.makedirs("patches",exist_ok=True)

    image=f"inputs/{basename}{ext}"
    basename=os.path.basename(image).replace(ext,'')
    image=load_image(image)#np.load(image)
    img_shape=image.shape[:-1]

    masks=dict()
    masks['tumor_map']=generate_tissue_mask(image,
                             compression=10,
                             otsu=False,
                             threshold=240,
                             connectivity=8,
                             kernel=5,
                             min_object_size=100000,
                             return_convex_hull=False,
                             keep_holes=False,
                             max_hole_size=6000,
                             gray_before_close=True,
                             blur_size=51)
    x_max,y_max=masks['tumor_map'].shape
    masks['macro_map']=fill_holes(masks['tumor_map'])

    patch_info=dict()
    for k in masks:
        patch_info[k]=pd.DataFrame([[basename,x,y,patch_size,"0"] for x,y in tqdm.tqdm(list(product(range(0,x_max-patch_size,patch_size),range(0,y_max-patch_size,patch_size))))],columns=['ID','x','y','patch_size','annotation'])
        patches=np.stack([image[x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())])
        include_patches=np.stack([masks[k][x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())]).mean((1,2))>=threshold

        np.save(f"masks/{basename}_{k}.npy",masks[k])
        np.save(f"patches/{basename}_{k}.npy",patches[include_patches])
        patch_info[k].iloc[include_patches].to_pickle(f"patches/{basename}_{k}.pkl")

    if secondary_patch_size:
        patch_info=dict()
        for k in ['tumor_map']:
            patch_info[k]=pd.DataFrame([[basename,x,y,secondary_patch_size,"0"] for x,y in tqdm.tqdm(list(product(range(0,x_max-secondary_patch_size,secondary_patch_size),range(0,y_max-secondary_patch_size,secondary_patch_size))))],columns=['ID','x','y','patch_size','annotation'])
            patches=np.stack([image[x:x+secondary_patch_size,y:y+secondary_patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())])
            include_patches=np.stack([masks[k][x:x+secondary_patch_size,y:y+secondary_patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())]).mean((1,2))>=threshold

            np.save(f"patches/{basename}_{k}_{secondary_patch_size}.npy",patches[include_patches])
            patch_info[k].iloc[include_patches].to_pickle(f"patches/{basename}_{k}_{secondary_patch_size}.pkl")
    return img_shape
