import pandas as pd, numpy as np
from scipy.ndimage import label as scilabel
from skimage.measure import regionprops_table
import cv2, os, subprocess
from deepzoom import *
from deepzoom import _get_or_create_path,_get_files_path
from PIL import Image
import tqdm
import dask
from dask.diagnostics import ProgressBar
from scipy.special import softmax
import torch
from sauth import SimpleHTTPAuthHandler, serve_http
from skimage.draw import circle
from .dzi_writer import Numpy2DZI
Image.MAX_IMAGE_PIXELS = None

colors=dict(red=np.array([255,0,0]),
           blue=np.array([0,0,255]),
           green=np.array([0,255,0]),
           yellow=np.array([255,255,0]))

@dask.delayed
def write_dzi(img, out_dzi, compression=8):
    return Numpy2DZI(compression=compression).create(img,out_dzi)

def add_depth(x):
    x=x.sort_values(['slide_id',"section_id"])
    x['depth']=np.arange(1,len(x)+1)
    return x

def mask2label(mask,compression=8):
    mask_small=cv2.resize(mask.astype(int),None,fx=1/compression,fy=1/compression,interpolation=cv2.INTER_NEAREST).astype(bool)
    label=cv2.resize(scilabel(mask_small)[0],dsize=mask.shape[::-1],interpolation=cv2.INTER_NEAREST)
    return label

class Case:
    def __init__(self, patient="163_A1"):
        self.launch_dir=os.path.abspath(".")
        self.patient=patient
        self.results=pd.read_pickle(f"results/{patient}.pkl")
        self.slide_metadata,self.section_metadata=self.add_metadata(self.results)
        self.n_slides=self.results['n_slides']
        self.slide_cache=self.slide_metadata.copy()
        self.section_cache=self.section_metadata.copy()
        self.slide_cache['label']=''
        self.slide_cache['tumor_map']=''
        self.slide_cache['macro_map']=''
        self.slide_cache['region_props']=''
        self.slide_inks={}
        self.section_metadata['quality']=''
        self.max_depth=self.section_metadata['depth'].max()
        self.n_blocks=self.section_metadata['block_id'].max()
        self.extraction_methods=dict(image=self.extract_section_image,
                                    tumor=self.extract_tumor_results,
                                    ink=self.extract_ink_results,
                                    nuclei=self.extract_nuclei_results)
        for k in self.extraction_methods.keys(): self.section_cache[f"{k}_dzi"]=''


    def add_metadata(self, results):
        slide_metadata=pd.DataFrame({k:results[k] for k in results if isinstance(results[k],list)}).reset_index().rename(columns=dict(index="slide_id"))
        slide_metadata['slide_id']+=1
        section_metadata=dict(slide_id=[],
                             block_id=[],
                             section_id=[],
                             label_id=[])
        for slide in slide_metadata['slide_id'].values:
            section_metadata['label_id'].extend(np.arange(1,results['n_sections_per_slide']*results['n_blocks_per_section']+1))
            section_metadata['slide_id'].extend([slide]*(results['n_sections_per_slide']*results['n_blocks_per_section']))
            section_metadata['block_id'].extend(np.arange(1,results['n_blocks_per_section']+1).tolist()*results['n_sections_per_slide'])
            for i in range(1,results['n_sections_per_slide']+1): section_metadata['section_id'].extend([i]*results['n_blocks_per_section'])
        section_metadata=pd.DataFrame(section_metadata)
        section_metadata['id']=np.arange(len(section_metadata))
        section_metadata=pd.DataFrame(section_metadata.groupby("block_id").apply(add_depth)).reset_index(drop=True).sort_values(['id'])
        return slide_metadata,section_metadata


    def compute_quality(self, importance_regions={'dermis':3.,'epidermis':1.,'subcutaneous tissue':2.},
                                importance_tumor=4.,
                                distance_weight=0.7,
                                baseline_region=1.):

        quality_scores=self.slide_metadata['quality_scores'].map(pd.read_pickle)

        self.tumor_quality_scores=pd.concat([quality_scores[i]['tumor'] for i in range(self.n_slides)],axis=1).fillna(0).T.reset_index(drop=True)
        self.macro_quality_scores=pd.concat([quality_scores[i]['macro'] for i in range(self.n_slides)],axis=1).fillna(0).T.reset_index(drop=True)

        self.section_metadata['quality']=np.nan
        for block_id in self.section_metadata['block_id'].unique():
            idx=self.section_metadata['block_id'].values==block_id
            section_metadata_ids=self.section_metadata['id'].loc[idx]
            macro_qual,tumor_qual=self.macro_quality_scores.loc[idx],self.tumor_quality_scores.loc[idx]
            for i in range(len(macro_qual)):
                macro_qual['distance_weight']=distance_weight**(np.abs(macro_qual.index-i))
                tumor_qual['distance_weight']=distance_weight**(np.abs(tumor_qual.index-i))
                quality_score=pd.concat([importance_regions[region]*(macro_qual[region]*(baseline_region+tumor_qual[region]*tumor_qual["distance_weight"])) for region in importance_regions]+[importance_tumor*tumor_qual['hole']*tumor_qual["distance_weight"]],axis=1)
                quality_score.columns=list(importance_regions)+['hole']
                self.section_metadata.loc[self.section_metadata['id']==section_metadata_ids[i],'quality']=quality_score.values.sum()

    def load_slide(self, slide, compression=8):
        slide_loc=self.get_slide_loc(slide)
        if self.slide_cache.loc[slide_loc,['images','masks','label']].map(lambda x: isinstance(x,str)).sum()>0:
            self.slide_cache.loc[slide_loc,['images','masks']]=self.slide_metadata.loc[slide_loc,['images','masks']].map(np.load)
            self.slide_cache.loc[slide_loc,'label']=[mask2label(self.slide_cache.loc[slide_loc,'masks'],compression)]
            self.slide_cache.loc[slide_loc,'region_props']=[regionprops_table(self.slide_cache.loc[slide_loc,'label'], properties=['bbox'])]
        image,mask,label=self.slide_cache.loc[slide_loc,['images','masks','label']].tolist()
        return image,mask,label

    def get_slide_loc(self, slide):
        return self.slide_metadata.index[np.where(self.slide_metadata['slide_id']==slide)[0][0]]

    def get_section_bbox(self, slide_id, label_id):
        slide_loc=self.get_slide_loc(slide_id)
        bbox=pd.DataFrame(self.slide_cache.loc[slide_loc,'region_props'])
        bbox.columns=['xmin','ymin','xmax','ymax']
        bbox.index+=1
        xmin,ymin,xmax,ymax=bbox.loc[label_id]
        return xmin,ymin,xmax,ymax

    def load_nuclei(self, slide):
        slide_loc=self.get_slide_loc(slide)
        if not isinstance(self.slide_cache.loc[slide_loc,'nuclei_results'],np.ndarray):
            nuclei_result=np.load(self.slide_cache.loc[slide_loc,'nuclei_results'])
            self.slide_cache.loc[slide_loc,'nuclei_results']=[nuclei_result]
        else: nuclei_result=self.slide_cache.loc[slide_loc,'nuclei_results']
        return nuclei_result

    def load_tumor_map(self, slide, alpha=0.1, patch_size=256, low_res=True):
        assert low_res, "High resolution label propagation completed, but not available as an option yet"
        slide_loc=self.get_slide_loc(slide)
        if not isinstance(self.slide_cache.loc[slide_loc,'tumor_gnn_results'],np.ndarray):
            graphs=torch.load(self.slide_cache.loc[slide_loc,'tumor_gnn_results'])
            xy=np.vstack([graph['xy'] for graph in graphs]).astype(int)
            y_pred=softmax(np.vstack([graph['y_pred'] for graph in graphs]),1)[:,1].reshape(-1,1)
            img_=self.load_slide(slide)[0].copy()
            one_square=np.ones((patch_size,patch_size)).astype(np.float)*255
            for x,y,pred in tqdm.tqdm(np.hstack([xy,y_pred]).tolist(), desc='tumor'):
                x,y=map(int,[x,y])
                img_[x:x+patch_size,y:y+patch_size]=alpha*cv2.applyColorMap(np.uint8(pred*one_square), cv2.COLORMAP_JET)+(1-alpha)*img_[x:x+patch_size,y:y+patch_size]
            self.slide_cache.loc[slide_loc,'tumor_gnn_results']=[img_]
        else: img_=self.slide_cache.loc[slide_loc,'tumor_gnn_results']
        return img_

    def load_ink(self, slide):
        slide_loc=self.get_slide_loc(slide)
        ink_file=self.slide_cache.loc[slide_loc,'ink_results']
        self.slide_inks.update({slide_loc:self.slide_inks.get(slide_loc,pd.read_pickle(ink_file))})
        return self.slide_inks[slide_loc]

    def extract_section_image(self, depth, block_id, compression=8):
        section=self.section_metadata.loc[(self.section_metadata['depth']==depth) & (self.section_metadata['block_id']==block_id)]
        label_id,slide_id=section.loc[:,['label_id','slide_id']].values.flatten()
        image,mask,label=self.load_slide(slide_id,compression)
        xmin,ymin,xmax,ymax=self.get_section_bbox(slide_id,label_id)

        img=image[xmin:xmax,ymin:ymax].copy()
        img[label[xmin:xmax,ymin:ymax]!=label_id]=255
        return img

    def extract_tumor_results(self, depth, block_id, alpha=0.3, patch_size=256, low_res=True, compression=8):
        section=self.section_metadata.loc[(self.section_metadata['depth']==depth) & (self.section_metadata['block_id']==block_id)]
        label_id,slide_id=section.loc[:,['label_id','slide_id']].values.flatten()
        _,mask,label=self.load_slide(slide_id,compression)
        xmin,ymin,xmax,ymax=self.get_section_bbox(slide_id,label_id)
        tumor_map=self.load_tumor_map(slide_id, alpha=alpha, patch_size=patch_size, low_res=low_res)

        img=tumor_map[xmin:xmax,ymin:ymax].copy()
        img[label[xmin:xmax,ymin:ymax]!=label_id]=255
        return img

    def extract_nuclei_results(self, depth, block_id, compression=8):
        section=self.section_metadata.loc[(self.section_metadata['depth']==depth) & (self.section_metadata['block_id']==block_id)]
        label_id,slide_id=section.loc[:,['label_id','slide_id']].values.flatten()
        image,mask,label=self.load_slide(slide_id,compression)
        xmin,ymin,xmax,ymax=self.get_section_bbox(slide_id,label_id)

        nuclei=self.load_nuclei(slide_id)

        img=image[xmin:xmax,ymin:ymax].copy()
        nuc_mask=nuclei[xmin:xmax,ymin:ymax].copy()
        img[nuc_mask,:]=[255,0,0]
        img[label[xmin:xmax,ymin:ymax]!=label_id]=255
        return img

    def extract_ink_results(self, depth, block_id, circle_size=200, compression=8):
        section=self.section_metadata.loc[(self.section_metadata['depth']==depth) & (self.section_metadata['block_id']==block_id)]
        label_id,slide_id=section.loc[:,['label_id','slide_id']].values.flatten()
        image,mask,label=self.load_slide(slide_id,compression)
        xmin,ymin,xmax,ymax=self.get_section_bbox(slide_id,label_id)
        xy_min=np.array([xmin,ymin])
        ink=self.load_ink(slide_id)[label_id]
        ink=ink.map(lambda x: (x-xy_min).astype(int))
        img=image[xmin:xmax,ymin:ymax].copy()
        max_size=np.array(img.shape[:2])
        for k in colors:
            if k!="center_mass":
                ink_k=ink.loc[k]
                remove=(~np.any((ink_k-max_size)>0,axis=1))
                ink_k=ink_k[remove]
                img[ink_k[:,0],ink_k[:,1],:]=colors[k]
            else:
                xx,yy=circle(*(ink.loc[k].astype(int).tolist()), circle_size)
                img[xx,yy,:]=[0,0,0]
        img[label[xmin:xmax,ymin:ymax]!=label_id]=255
        return img

    def write_dzi(self, img, out_dzi, compression=8):
        Numpy2DZI(compression=compression).create(img,out_dzi)

    def write_dzi_parallel(self, img_dzi_dict, compression=8, scheduler='processes'):
        written_dzis=[]
        for out_dzi, img in img_dzi_dict.items():
            written_dzis.append(write_dzi(img, out_dzi, compression))
        with ProgressBar():
            written_dzis=dask.compute(*written_dzis, scheduler=scheduler)
        return written_dzis

    def launch_server(self, username='username', password='password', port=5554):
        self.reset_dir()
        SimpleHTTPAuthHandler.username = username
        SimpleHTTPAuthHandler.password = password
        serve_http(ip='localhost', port=port, https=False,
               start_dir='dzi_files', handler_class=SimpleHTTPAuthHandler)

    def reset_dir(self):
        os.chdir(self.launch_dir)

    def visualize_dzi(self, dzis):
        self.reset_dir()
        replace_txt='","'.join(list(map(os.path.basename,dzis)))
        with open("osd_template.html") as f_in, open('dzi_files/index.html','w') as f_out:
            f_out.write(f_in.read().replace("REPLACE",replace_txt).replace("BASENAME",self.patient))

    def extract2dzi(self, image_type='image', scheduler='single-threaded'):
        assert image_type in ['image','nuclei','tumor','ink']
        os.makedirs("dzi_files",exist_ok=True)
        dzi_files=[]
        imgs={}
        for block in tqdm.trange(1,self.n_blocks+1, desc='block'):
            for depth in tqdm.trange(1,self.max_depth+1, desc='depth'):
                section_info=[self.patient,depth,block,image_type]
                dzi_file=f"dzi_files/{'_'.join(list(map(str,section_info)))}.dzi"
                dzi_files.append(dzi_file)
                if not os.path.exists(dzi_file):
                    imgs[dzi_file]=self.extraction_methods[image_type](depth,block)
        if len(imgs)>0: self.write_dzi_parallel(imgs,scheduler=scheduler)
        self.section_cache[f"{image_type}_dzi"]=self.section_cache[['depth','block_id']].apply(lambda x: f"dzi_files/{'_'.join([self.patient]+x.values.astype(str).tolist()+[image_type])}.dzi",axis=1)
        return dzi_files
