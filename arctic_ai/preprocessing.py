import os, tqdm
import numpy as np, pandas as pd
from pathflowai.utils import generate_tissue_mask
from itertools import product
from scipy.ndimage.morphology import binary_fill_holes as fill_holes
from pathpretrain.utils import load_image

def preprocess(basename="163_A1a",
               threshold=0.05,
               patch_size=256,
               ext='.npy'):

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
    return img_shape
