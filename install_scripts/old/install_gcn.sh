#python version: 3.6

# c/c++ compilers:
# need > 5 and < 10
conda install -y gcc_linux-64
conda install -y gxx_linux-64
# https://seanlaw.github.io/2019/01/17/pip-installing-wheels-with-conda-gcc/
# export CC=/dartfs-hpc/rc/home/9/f005q19/anaconda3/envs/{name}/bin/x86_64-conda_cos6-linux-gnu-gcc
# export CXX=/dartfs-hpc/rc/home/9/f005q19/anaconda3/envs/{name}/bin/x86_64-conda_cos6-linux-gnu-g++

conda install -y pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -y opencv
pip install -y histomicstk --find-links https://girder.github.io/large_image_wheels
pip install -y git+https://github.com/jlevy44/PathPretrain

yes | pip install fire
yes | pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd
yes | pip install seaborn

#gcn:
#https://github.com/rusty1s/pytorch_scatter/issues/245
yes | pip install torch-scatter==2.0.7 torch-sparse==0.6.12 -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
yes | pip install torch-cluster torch-spline-conv -f https://pytorch-geometric.com/whl/torch-torch-1.7.1+cu110.html
yes | pip install torch-geometric 

#jupyter/slurm:
yes | pip install ipykernel
yes | pip install ipywidgets
yes | pip install git+https://github.com/jlevy44/Submit-HPC