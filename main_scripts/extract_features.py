import fire,pickle,subprocess
import numpy  as np

def launch_job(ID="",patch_size=256,out_dir="imagenet_embeddings/"):
    gpu=np.random.randint(4)
    input_dir="bcc"
    subprocess.call(f"export PATH=/dartfs-hpc/rc/home/w/f003k8w/.local/bin:/dartfs-hpc/rc/home/w/f003k8w/.local/lib/python3.7/site-packages/:$PATH && cd {input_dir} && pathflowai-train_model train_model -gpu {gpu} --npy_file inputs/{ID}.npy  --prediction_output_dir {out_dir} -ee --prediction -olf bce -ca --patch_size {patch_size}  --pos_annotation_class Cancer -oa Benign  -pr 224 --save_location .pkl -a resnet18 --input_dir inputs/ -bs 32 -nt 1 --mt_bce -lr 1e-4 -ne 25 -pi patch_info_update.db",shell=True)
    
if __name__=="__main__":
    fire.Fire(launch_job)