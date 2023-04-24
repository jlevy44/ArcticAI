"""
CNN 
==========

Contains functions related to generating embeddings for image patches using a convolutional neural network
"""
import os, torch, tqdm, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms
import warnings

class CustomDataset(Dataset):
    # load using saved patches and mask file
    def __init__(self, ID, patch_info, X, transform):
        self.X=X
        self.patch_info=patch_info
        self.xy=self.patch_info[['x','y']].values
        self.patch_size=self.patch_info['patch_size'].iloc[0]
        self.length=self.patch_info.shape[0]
        self.transform=transform
        self.to_pil=lambda x: Image.fromarray(x)
        self.ID=ID#os.path.basename(npy_file).replace(".npy","")

    def __getitem__(self,i):
        x,y=self.xy[i]
        return self.transform(self.to_pil(self.X[i]))#[x:x+patch_size,y:y+patch_size]

    def __len__(self):
        return self.length

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,X in tqdm.tqdm(enumerate(dataloader),total=n_batches):
                if torch.cuda.is_available(): X=X.cuda()
                z=model(X).detach().cpu().numpy()
                Z.append(z)
        Z=np.vstack(Z)
        torch.save(dict(embeddings=Z,patch_info=self.patch_info),os.path.join(out_dir,f"{self.ID}.pkl"))

class CustomDatasetOld(Dataset):
    # load using saved patches and mask file
    def __init__(self, patch_info, npy_file, transform):
        warnings.warn(
                "This dataset class is deprecated.",
                DeprecationWarning
            )
        raise RuntimeError
        self.X=np.load(npy_file)
        self.patch_info=pd.read_pickle(patch_info)
        self.xy=self.patch_info[['x','y']].values
        self.patch_size=self.patch_info['patch_size'].iloc[0]
        self.length=self.patch_info.shape[0]
        self.transform=transform
        self.to_pil=lambda x: Image.fromarray(x)
        self.ID=os.path.basename(npy_file).replace(".npy","")

    def __getitem__(self,i):
        x,y=self.xy[i]
        return self.transform(self.to_pil(self.X[i]))#[x:x+patch_size,y:y+patch_size]

    def __len__(self):
        return self.length

    def embed(self,model,batch_size,out_dir):
        Z=[]
        dataloader=DataLoader(self,batch_size=batch_size,shuffle=False)
        n_batches=len(self)//batch_size
        with torch.no_grad():
            for i,X in tqdm.tqdm(enumerate(dataloader),total=n_batches):
                if torch.cuda.is_available(): X=X.cuda()
                z=model(X).detach().cpu().numpy()
                Z.append(z)
        Z=np.vstack(Z)
        torch.save(dict(embeddings=Z,patch_info=self.patch_info),os.path.join(out_dir,f"{self.ID}.pkl"))

def generate_embeddings(basename="163_A1a",
                        analysis_type="tumor",
                       gpu_id=0,
                       dirname="."):
    """ 
    Generate embeddings for patches in a WSI.

    Parameters
    ----------
    basename : str
        Basename of the WSI.
    analysis_type : str
        Type of analysis to perform. Can be either "tumor" or "macro".
    gpu_id : int, optional
        GPU to use for training. If not provided, uses CPU.
    dirname : str, optional
        Directory containing data for the WSI.

    Returns
    -------
    None
        The function saves the generated embeddings to the `cnn_embeddings` directory.
    """

    os.makedirs(os.path.join(dirname,"cnn_embeddings"),exist_ok=True)

    patch_info_file,npy_file=os.path.join(dirname,f"patches/{basename}.pkl"),os.path.join(dirname,f"patches/{basename}.npy")
    models={k:os.path.join(dirname,f"models/{k}_map_cnn.pth") for k in ['macro','tumor']}
    num_classes=dict(macro=4,tumor=3)

    npy_stack=np.load(npy_file)
    patch_info=pd.read_pickle(patch_info_file)
    if f"{analysis_type}_map" in patch_info.columns:
        npy_stack=npy_stack[patch_info[f"{analysis_type}_map"].values]
        patch_info=patch_info[patch_info[f"{analysis_type}_map"].values]
    train_model(model_save_loc=models[analysis_type],extract_embeddings=True,num_classes=num_classes[analysis_type],predict=True,embedding_out_dir=os.path.join(dirname,"cnn_embeddings/"),custom_dataset=CustomDataset(f"{basename}_{analysis_type}_map",patch_info,npy_stack,generate_transformers(224,256)['test']),gpu_id=gpu_id)


def generate_embeddings_old(basename="163_A1a",
                        analysis_type="tumor",
                       gpu_id=0,
                       dirname="."):

    warnings.warn(
            "Old generate embeddings function is deprecated",
            DeprecationWarning
        )
    raise RuntimeError

    os.makedirs(os.path.join(dirname,"cnn_embeddings"),exist_ok=True)

    patch_info_file,npy_file=os.path.join(dirname,f"patches/{basename}_{analysis_type}_map.pkl"),os.path.join(dirname,f"patches/{basename}_{analysis_type}_map.npy")
    models={k:f"models/{k}_map_cnn.pth" for k in ['macro','tumor']}
    num_classes=dict(macro=4,tumor=3)
    train_model(model_save_loc=models[analysis_type],extract_embeddings=True,num_classes=num_classes[analysis_type],predict=True,embedding_out_dir=os.path.join(dirname,"cnn_embeddings/"),custom_dataset=CustomDataset(patch_info_file,npy_file,generate_transformers(224,256)['test']),gpu_id=gpu_id)
