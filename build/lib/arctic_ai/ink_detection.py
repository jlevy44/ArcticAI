"""
Ink Detection 
==========

Detect inks for tissue orientation.
"""
from skimage import morphology as morph
from scipy.ndimage import binary_opening, binary_dilation, label as scilabel
from skimage import filters, measure
from skimage.morphology import disk
import numpy as np, pandas as pd, copy
import sys,os,cv2
from itertools import product
from pathpretrain.utils import load_image
import warnings
import dask
from dask.diagnostics import ProgressBar
# sys.path.insert(0,os.path.abspath('.'))
from .filters import filter_red_pen, filter_blue_pen, filter_green_pen

def filter_yellow(img): # https://www.learnopencv.com/color-spaces-in-opencv-cpp-python/
    img_hsv=cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return cv2.inRange(img_hsv,(10, 30, 30), (30, 255, 255))

ink_fn=dict(red=filter_red_pen,
           blue=filter_blue_pen,
           green=filter_green_pen,
           yellow=filter_yellow)

ink_min_size=dict(red=100,
           blue=30,
           green=30,
           yellow=1000)

colors=dict(red=np.array([255,0,0]),
           blue=np.array([0,0,255]),
           green=np.array([0,255,0]),
           yellow=np.array([255,255,0]))

def tune_mask(mask,edges,min_size=30):
    mask=(binary_dilation(mask,disk(3,bool),iterations=5) & edges)
    mask=binary_opening(mask,disk(3,bool),iterations=1)
    return morph.remove_small_objects(mask, min_size=min_size, connectivity = 2, in_place=True)>0

def filter_tune(img,color,edges):
    return tune_mask(~ink_fn[color](img),edges,min_size=ink_min_size[color])

def get_edges(mask):
    edges=filters.sobel(mask)>0
    edges = binary_dilation(edges,disk(30,bool))
    return edges

def detect_inks(basename="163_A1a",
                compression=8.,
                mask_compressed=True,
                ext=".npy",
                dirname=".",
                return_mean=False):
    """
    Detect inks in the specified image.

    Parameters
    ----------
    basename : str, optional
        The base name of the image to detect inks in, by default "163_A1a"
    compression : float, optional
        The compression factor to use when detecting inks, by default 8.
    mask_compressed : bool, optional
        Whether or not to use compressed masks, by default True
    ext : str, optional
        The file extension of the image, by default ".npy"
    dirname : str, optional
        The directory containing the image and masks, by default "."
    return_mean : bool, optional
        Whether or not to return the mean location of the detected inks, by default False

    Returns
    -------
    None
        The function saves the detected inks to a pickle file, but does not return anything.
    """

    os.makedirs(os.path.join(dirname,"detected_inks"),exist_ok=True)

    img=load_image(f"{dirname}/inputs/{basename}{ext}")
    xy_bounds=pd.read_pickle(os.path.join(dirname,"masks",f"{basename}.pkl"))
    with ProgressBar():
        masks=dask.compute({ID:dask.delayed(np.load)(f"{dirname}/masks/{basename}_{ID}.npy") for ID in xy_bounds},scheduler="threading")[0]
    pen_masks=dict()
    coord_translate={}
    for ID in xy_bounds:
        (xmin,ymin),(xmax,ymax)=xy_bounds[ID]
        msk=masks[ID].astype(np.uint8)
        if not mask_compressed: msk=cv2.resize(msk.astype(int),None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_NEAREST).astype(bool)
        coord_translate[ID]=np.array([xmin,ymin])
        im=cv2.resize(img[xmin:xmax,ymin:ymax],None,fx=1/compression,fy=1/compression)
        edges=dask.delayed(get_edges)(msk)#)
        com=np.vstack(np.where(msk)).T.mean(0)*compression
        pen_masks[ID]={color:dask.delayed(lambda x,y,z,k: np.vstack(np.where(z & filter_tune(x,k,y))).T*compression)(im,edges,msk,color) for color in ink_fn}
        pen_masks[ID]['center_mass']=com
    with ProgressBar():
        pen_masks=dask.compute(pen_masks,scheduler="threading")[0]
    pd.to_pickle(pen_masks,f"{dirname}/detected_inks/{basename}.pkl")
    return None


def detect_inks_old_v3(basename="163_A1a",
                compression=8,
                mask_compressed=True,
                ext=".npy",
                dirname=".",
                return_mean=False):

    os.makedirs(os.path.join(dirname,"detected_inks"),exist_ok=True)

    img=load_image(f"{dirname}/inputs/{basename}{ext}")
    xy_bounds=pd.read_pickle(os.path.join(dirname,"masks",f"{basename}.pkl"))
    img=cv2.resize(img,None,fx=1/compression,fy=1/compression)
    mask=np.zeros(img.shape[:-1],dtype=np.uint8)
    with ProgressBar():
        masks=dask.compute({ID:dask.delayed(np.load)(f"{dirname}/masks/{basename}_{ID}.npy") for ID in xy_bounds},scheduler="threading")[0]
    coord_translate={}
    for ID in xy_bounds:
        (xmin,ymin),(xmax,ymax)=xy_bounds[ID]
        coord_translate[ID+1]=np.array([xmin,ymin])
        msk=masks[ID].astype(np.uint8)#np.load(f"{dirname}/masks/{basename}_{ID}.npy").astype(np.uint8)
        if mask_compressed:
            xmin,ymin,xmax,ymax=(np.array([xmin,ymin,xmax,ymax])/compression).astype(int)
            xmax,ymax=np.array([xmin,ymin]+np.array(msk.shape))
        msk[msk>0]=ID+1
        mask[xmin:xmax,ymin:ymax]=msk
    if not mask_compressed: mask=cv2.resize(mask.astype(int),None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_NEAREST).astype(bool)
    labels,mask=mask,mask>0
    n_objects=np.max(labels.flatten())
    # labels,n_objects=scilabel(mask)
    edges=get_edges(mask)
    pen_masks={k:filter_tune(img,k,edges) for k in ink_fn}

    # for k in ['green','blue','red','yellow']:
    #     img[pen_masks[k],:]=colors[k]

    coords_df=pd.DataFrame(index=list(ink_fn.keys())+["center_mass"],columns=np.arange(1,n_objects+1))#
    for color,obj in product(coords_df.index[:-1],coords_df.columns):
        coords_df.loc[color,obj]=np.vstack(np.where((labels==obj) & (pen_masks[color]))).T*compression-coord_translate[obj]
        if return_mean: coords_df.loc[color,obj]=coords_df.loc[color,obj].mean(0)
    for obj in coords_df.columns:
        coords_df.loc["center_mass",obj]=np.vstack(np.where(labels==obj)).T.mean(0)*compression-coord_translate[obj]

    coords_df.to_pickle(f"{dirname}/detected_inks/{basename}.pkl")

def detect_inks_old_v2(basename="163_A1a_1",compression=8.,*args,**kwargs):

    os.makedirs("detected_inks",exist_ok=True)

    img=load_image(f"inputs/{basename}.npy")
    mask=np.load(f"masks/{basename}.npy")
    ID=int(basename.split("_")[-1])
    labels,mask=mask,labels>0
    edges=get_edges(mask)
    pen_masks={k:filter_tune(img,k,edges) for k in ink_fn}

    # for k in ['green','blue','red','yellow']:
    #     img[pen_masks[k],:]=colors[k]

    coords_df=pd.DataFrame(index=list(ink_fn.keys())+["center_mass"],columns=[ID])#
    coords_df.loc[color,ID]=np.vstack(np.where(mask & (pen_masks[color]))).T*compression
    coords_df.loc["center_mass",ID]=np.vstack(np.where(mask)).T.mean(0)*compression

    coords_df.to_pickle(f"detected_inks/{basename}.pkl")

def detect_inks_old(basename="163_A1a",
                compression=8,
                ext=".npy"):

    warnings.warn(
            "Old ink detection is deprecated",
            DeprecationWarning
        )
    raise RuntimeError

    os.makedirs("detected_inks",exist_ok=True)

    img,mask=load_image(f"inputs/{basename}{ext}"),np.load(f"masks/{basename}_macro_map.npy")
    img=cv2.resize(img,None,fx=1/compression,fy=1/compression)
    mask=cv2.resize(mask.astype(int),None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_NEAREST).astype(bool)
    labels,n_objects=scilabel(mask)
    edges=get_edges(mask)
    pen_masks={k:filter_tune(img,k,edges) for k in ink_fn}

    for k in ['green','blue','red','yellow']:
        img[pen_masks[k],:]=colors[k]

    coords_df=pd.DataFrame(index=list(ink_fn.keys())+["center_mass"],columns=np.arange(1,n_objects+1))
    for color,obj in product(coords_df.index[:-1],coords_df.columns):
        coords_df.loc[color,obj]=np.vstack(np.where((labels==obj) & (pen_masks[color]))).T*compression
    for obj in coords_df.columns:
        coords_df.loc["center_mass",obj]=np.vstack(np.where(labels==obj)).T.mean(0)*compression

    coords_df.to_pickle(f"detected_inks/{basename}.pkl")
    np.save(f"detected_inks/{basename}_thumbnail.npy",img)
