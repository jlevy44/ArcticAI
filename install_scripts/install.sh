#python version: 3.6

# c/c++ compilers:
# need > 5 and < 10
conda install gcc_linux-64
conda install gxx_linux-64
# https://seanlaw.github.io/2019/01/17/pip-installing-wheels-with-conda-gcc/
# export CC=/dartfs-hpc/rc/home/9/f005q19/anaconda3/envs/{name}/bin/x86_64-conda_cos6-linux-gnu-gcc
# export CXX=/dartfs-hpc/rc/home/9/f005q19/anaconda3/envs/{name}/bin/x86_64-conda_cos6-linux-gnu-g++

conda install -c defaults -c conda-forge -c anaconda tqdm pandas networkx openslide scipy dask fire scikit-learn tifffile
# pip install opencv-python
conda install opencv
pip install histomicstk --find-links https://girder.github.io/large_image_wheels
pip install git+https://github.com/jlevy44/PathPretrain
pip install ==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
# python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install fire
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
pip install seaborn

#cpc:
pip install lightning-bolts
pip install umap-learn
pip install 'ray[default]'
pip install xmltodict
pip install git+https://github.com/ray-project/ray_lightning 

#gcn:
#https://github.com/rusty1s/pytorch_scatter/issues/245
pip install torch-scatter==2.0.7 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-torch-1.7.1+cu110.html
pip install torch-geometric 

#jupyter/slurm:
pip install ipykernel
pip install ipywidgets
pip install git+https://github.com/jlevy44/Submit-HPC

#unet/crf:
pip install wandb
pip install git+https://github.com/lucasb-eyer/pydensecrf.git