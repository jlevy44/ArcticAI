import os, tqdm, glob
import fire, tifffile
import numpy as np, pandas as pd
from pathflowai.utils import generate_tissue_mask
from itertools import product
from scipy.ndimage.morphology import binary_fill_holes as fill_holes

def extract_patches(basename="163_A1a",
                    input_dir='inputs',
                    output_dir='patch_info',
               threshold=0.05,
               patch_size=256):
    
    image=(glob.glob(f"{input_dir}/{basename}*.npy")+glob.glob(f"{input_dir}/{basename}*.tif"))[0]
    basename=os.path.basename(image).replace('.npy','').replace('.tif','').replace("_ASAP","")
    image=np.load(image) if image.endswith(".npy") else tifffile.imread(image)
    
    masks=dict()
    masks['tumor']=generate_tissue_mask(image,
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
    x_max,y_max=masks['tumor'].shape
    masks['macro']=fill_holes(masks['tumor'])
    patch_info=dict()
    for k in masks:
        patch_info[k]=pd.DataFrame([[basename,x,y,patch_size,"0"] for x,y in tqdm.tqdm(list(product(range(0,x_max-patch_size,patch_size),range(0,y_max-patch_size,patch_size))))],columns=['ID','x','y','patch_size','annotation'])
        include_patches=np.stack([masks[k][x:x+patch_size,y:y+patch_size] for x,y in tqdm.tqdm(patch_info[k][['x','y']].values.tolist())]).mean((1,2))>=threshold
        patch_info[k].iloc[include_patches].to_pickle(f"{output_dir}/{k}/{basename}.pkl")
        
if __name__=="__main__":
    fire.Fire(extract_patches)