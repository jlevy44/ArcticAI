import os, torch, tqdm, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathpretrain.train_model import train_model, generate_transformers, generate_kornia_transforms

class CustomDataset(Dataset):
    # load using saved patches and mask file
    def __init__(self, patch_info, npy_file, transform):
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
                       gpu_id=0):
    patch_info_file,npy_file=f"patches/{basename}_{analysis_type}_map.pkl",f"patches/{basename}_{analysis_type}_map.npy"
    models={k:f"models/{k}_map_cnn.pth" for k in ['macro','tumor']}
    num_classes=dict(macro=4,tumor=3)
    train_model(model_save_loc=models[analysis_type],extract_embeddings=True,num_classes=num_classes[analysis_type],predict=True,embedding_out_dir="cnn_embeddings/",custom_dataset=CustomDataset(patch_info_file,npy_file,generate_transformers(224,256)['test']),gpu_id=gpu_id)