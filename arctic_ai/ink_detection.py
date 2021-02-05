from skimage import morphology as morph
from scipy.ndimage import binary_opening, binary_dilation, label as scilabel
from skimage import filters, measure
from skimage.morphology import disk
import numpy as np, pandas as pd, copy
import sys,os,cv2
from itertools import product
sys.path.insert(0,os.path.abspath('.'))
from filters import filter_red_pen, filter_blue_pen, filter_green_pen

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
                compression=8):
    img,mask=np.load(f"inputs/{basename}.npy"),np.load(f"masks/{basename}_macro_map.npy")
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
