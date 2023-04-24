"""Contains functions for detecting follicles in images."""

from skimage.draw import disk
from PIL import Image
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
import os
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch
import tifffile
import tqdm
import time
from scipy.special import softmax
import dask
from dask.diagnostics import ProgressBar
from pathpretrain.utils import load_image
import fire

white_square=np.ones((1024,1024,3),dtype=np.uint8)*255

def check_update(new_square):
    white_square_=new_square
    if not np.all(np.array(new_square.shape)==np.array((1024,1024,3))):
        white_square_=white_square.copy()
        white_square_[:new_square.shape[0],:new_square.shape[1]]=new_square
    return white_square_

def update_gnn_res(out_fname,gnn_res,alpha,patches_new,disk_mask,y_pred,y_tumor,tumor_thres,model,batch_size,num_workers):
    X=TensorDataset(torch.FloatTensor(patches_new[...,::-1].copy().astype("float32")).permute(0,3,1,2))#/255
    dataloader=DataLoader(X,shuffle=False,batch_size=batch_size,num_workers=num_workers)
    preds=[]
    with torch.no_grad():
        for x, in dataloader:#tqdm.tqdm(,total=len(dataloader.dataset)//dataloader.batch_size):
            preds.append(np.stack([y_pred['panoptic_seg'][0].cpu().numpy() for y_pred in model([{"image":im, "height":1024, "width":1024} for im in x.cuda()])]))
        preds=np.concatenate(preds,0)
    y_tumor_new=[max(y_tumor_pred-np.mean([alpha[j-1]*(follicle_pred[disk_mask==j]>0).mean() for j in range(1,4)]),0) for y_tumor_pred, follicle_pred in zip(y_tumor[y_tumor>tumor_thres],preds)] 
    y_benign=y_pred[:,0].copy()
    y_benign[y_tumor>tumor_thres]=y_benign[y_tumor>tumor_thres]+y_pred[y_tumor>tumor_thres,2]-y_tumor_new#verify, add, update with inverse 
    y_tumor[y_tumor>tumor_thres]=y_tumor_new
    gnn_res['y_pred_orig']=y_pred.copy()
    gnn_res['y_pred']=y_pred.copy()
    gnn_res['y_pred'][:,0]=y_benign 
    gnn_res['y_pred'][:,2]=y_tumor
    torch.save([gnn_res],out_fname)
    return None

def predict_hair_follicles(tumor_thres=0.3,
            patch_size=256,
            basename="340_A1a_ASAP",
            alpha_scale=2.,
            alpha=[1.,2.,3.],
            model_path="model_final.pth",
            model_dir="./output",
            detectron_threshold=0.55,
            dirname="../../bcc_test_set/",
            ext=".tif",
            batch_size=16,
            num_workers=1):
    """
    Predict hair follicles in a given tumor image using a pre-trained GNN model.

    Args:
    - tumor_thres (float): Threshold value for the tumor prediction probability (default 0.3).
    - patch_size (int): Size of the image patches used for prediction (default 256).
    - basename (str): Name of the input image file without the extension (default "340_A1a_ASAP").
    - alpha_scale (float): Scaling factor for the alpha values used in GNN, reduces the tumor prediction probability (default 2.0).
    - alpha (list): List of 3 alpha values used in GNN to reduce tumor probability in presence of follicles (default [1., 2., 3.]).
    - model_path (str): Path to the pre-trained GNN model file (default "model_final.pth").
    - model_dir (str): Directory path to save the GNN model (default "./output").
    - detectron_threshold (float): Threshold value for object detection (default 0.55).
    - dirname (str): Directory path to the input image file (default "../../bcc_test_set/").
    - ext (str): Extension of the input image file (default ".tif").
    - batch_size (int): Batch size used for prediction (default 16).
    - num_workers (int): Number of worker processes for data loading (default 1).

    Returns:
    - None.
    """
    
    threshold=detectron_threshold
    base_model="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"
    n=1
    os.makedirs(os.path.join(dirname,"gnn_follicle_results"),exist_ok=True)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(base_model))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = n
    cfg.OUTPUT_DIR=model_dir
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, model_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    predictor = DefaultPredictor(cfg)
    model=predictor.model
    alpha=np.array(alpha)/sum(alpha)*alpha_scale
    d_patch=int(patch_size/2)
    disk_mask=np.zeros((1024,1024))
    xx,yy=disk((512,512),512)
    disk_mask[xx,yy]=1
    xx,yy=disk((512,512),256)
    disk_mask[xx,yy]=2
    xx,yy=disk((512,512),128)
    disk_mask[xx,yy]=3

    img_=load_image(os.path.join(dirname,"inputs",f"{basename}{ext}"))

    gnn_res_={}
    for bn in pd.read_pickle(os.path.join(dirname,"metadata",f"{basename}.pkl")):
        patch_info=pd.read_pickle(os.path.join(dirname,"patches",f"{bn}.pkl"))
        patches=np.load(os.path.join(dirname,"patches",f"{bn}.npy"))[patch_info['tumor_map'].values]
        patch_info=patch_info[patch_info['tumor_map'].values]

        gnn_res=torch.load(os.path.join(dirname,"gnn_results",f"{bn}_tumor_map.pkl"))[0]
        y_pred=softmax(gnn_res['y_pred'],1).copy()
        y_tumor=y_pred[:,2].copy()
        patches_new=[check_update(img_[x+d_patch-512:x+d_patch+512,y+d_patch-512:y+d_patch+512]) for x,y in tqdm.tqdm(patch_info[['x_orig','y_orig']][y_tumor>tumor_thres].values.tolist())]
        if len(patches_new): 
            patches_new=np.stack(patches_new)
            gnn_res_[bn]=dask.delayed(update_gnn_res)(os.path.join(dirname,"gnn_follicle_results",f"{bn}_tumor_map.pkl"),gnn_res,alpha,patches_new,disk_mask,y_pred,y_tumor,tumor_thres,model,batch_size,num_workers)
        else:
            gnn_res_[bn]=[gnn_res]
    with ProgressBar():
        gnn_res_=dask.compute(gnn_res_,scheduler="threading")[0]

        
if __name__=="__main__":
    fire.Fire(predict_hair_follicles)