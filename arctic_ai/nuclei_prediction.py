from PIL import Image
from torch.utils.data import Dataset
import torch, pandas as pd, numpy as np
import pickle, os
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms
from tqdm import trange
from scipy.special import softmax
import warnings

class WSI_Dataset(Dataset):
    def __init__(self, patches, transform):
        self.patches=patches
        self.to_pil=lambda x: Image.fromarray(x)
        self.length=len(self.patches)
        self.transform=transform

    def __getitem__(self,idx):
        X=self.transform(self.to_pil(self.patches[idx]))
        return X,torch.zeros(X.shape[-2:]).unsqueeze(0).long()

    def __len__(self):
        return self.length

def predict_nuclei(basename="163_A1a",
                   gpu_id=0,
                   return_prob=False,
                   ext=".npy",
                   img_shape=None):
    warnings.warn(
            "Nuclei prediction is deprecated for Detectron Module",
            DeprecationWarning
        )
    raise RuntimeError

    os.makedirs("nuclei_results",exist_ok=True)

    analysis_type="tumor"
    patch_size=256
    patch_info_file,npy_file=f"patches/{basename}_{analysis_type}_map.pkl",f"patches/{basename}_{analysis_type}_map.npy"
    patches=np.load(npy_file)
    custom_dataset=WSI_Dataset(patches,generate_transformers(256,256)['test'])
    Y_seg=train_model(inputs_dir='inputs',
                    architecture='resnet50',
                    batch_size=512,
                    num_classes=2,
                    predict=True,
                    model_save_loc="models/nuclei.pth",
                    predictions_save_path='tmp_test.pkl',
                    predict_set='custom',
                    verbose=False,
                    class_balance=False,
                    gpu_id=gpu_id,
                    tensor_dataset=False,
                    semantic_segmentation=True,
                    custom_dataset=custom_dataset,
                    save_predictions=False)['pred']

    xy=pd.read_pickle(patch_info_file)[['x','y']].values
    if ext==".npy":
        img_shape=np.load(f"inputs/{basename}.npy",mmap_mode="r").shape[:-1]
    elif isinstance(img_shape,type(None)):
        img_shape=load_image(f"inputs/{basename}{ext}").shape[:-1]
    else:
        assert not isinstance(img_shape,type(None))
    pred_mask=np.zeros(img_shape)
    for i in trange(Y_seg.shape[0]):
        x,y=xy[i]
        y_prob=Y_seg[i]
        pred_mask[x:x+patch_size,y:y+patch_size]=y_prob.argmax(0) if not return_prob else softmax(y_prob,axis=0)[1,...]
    if not return_prob: pred_mask=pred_mask.astype(bool)
    np.save(f"nuclei_results/{basename}.npy",pred_mask)
