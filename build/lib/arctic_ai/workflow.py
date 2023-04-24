"""Serial Workflow 
==========
Contains functions for serial processing of tissue sections. 
Contains functions for defining and running workflows."""
import glob, os, time, pickle
from .preprocessing import preprocess
from .cnn_prediction import generate_embeddings
from .generate_graph import create_graph_data
from .gnn_prediction import predict
# from .quality_scores import generate_quality_scores
from .ink_detection import detect_inks
#from .compile_results import dump_results
#from .nuclei_detection import predict_nuclei
from .image_stitch import npy2dzi, stitch_slides
#from .case_prototype import Case
import warnings
from functools import partial
import pandas as pd

def files_exist_overwrite(overwrite, files):
    return (not overwrite) and all([os.path.exists(file) for file in files])

def generate_output_file_names(basename):
    out_files=dict()
    out_files['preprocess']=[f"masks/{basename}_{k}_map.npy" for k in ['tumor','macro']]+[f"patches/{basename}_{k}_map.npy" for k in ['tumor','macro']]
    for k in ['macro','tumor']:
        out_files[f'cnn_{k}']=[f"cnn_embeddings/{basename}_{k}_map.pkl"]
        out_files[f'graph_data_{k}']=[f"graph_datasets/{basename}_{k}_map.pkl"]
        out_files[f'gnn_{k}']=[f"gnn_results/{basename}_{k}_map.pkl"]
    out_files['quality']=[f"quality_scores/{basename}.pkl"]
    out_files['ink']=[f"detected_inks/{basename}_thumbnail.npy"]
    out_files['nuclei']=[f"nuclei_results/{basename}.npy"]
    return out_files

def run_workflow_series(basename, compression, overwrite, ext, dirname, df_section_pieces_file, run_stitch_slide):
    """
    Runs image processing workflow in series on an input image.

    Parameters
    ----------
    basename : str
        The base name of the slide to process.
    compression : float
        The level of compression to apply to the slide.
    overwrite : bool
        Whether to overwrite existing files if they exist.
    ext : str
        The file extension of the slide.
    dirname : str
        The directory containing the slide and other relevant files.
    df_section_pieces_file : str
        The file containing information about the patches.
    run_stitch_slide : bool
        Whether to run the stitch_slides function after all other processing is complete.

    Returns
    -------
    times : dict
        A dictionary containing the times at which each step of the workflow was completed.
    """

    print(f"{basename} preprocessing")

    out_files=generate_output_file_names(basename)

    times=dict()
    times['start']=time.time()

    if not files_exist_overwrite(overwrite,out_files['preprocess']):
        preprocess(basename=basename,
               threshold=0.05,
               patch_size=256,
               ext=ext,
               no_break=False,
               df_section_pieces_file=df_section_pieces_file,
               image_mask_compression=8.,
               dirname=dirname)

    times['preprocess']=time.time()

    new_basenames=pd.read_pickle(os.path.join(dirname,"metadata",f"{basename}.pkl"))

    for k in ['tumor','macro']: times[f'cnn_{k}'],times[f'graph_{k}'],times[f'gnn_{k}']=[],[],[]
    for bn in new_basenames:
        for k in ['tumor','macro']:
            print(f"{bn} {k} embedding")
            if not files_exist_overwrite(overwrite,out_files[f'cnn_{k}']):
                generate_embeddings(basename=bn,
                                analysis_type=k,
                               gpu_id=-1,
                               dirname=dirname)
            times[f'cnn_{k}'].append(time.time())

            print(f"{bn} {k} build graph")
            if not files_exist_overwrite(overwrite,out_files[f'graph_data_{k}']):
                create_graph_data(basename=bn,
                              analysis_type=k,
                              radius=256,
                              min_component_size=600,
                              no_component_break=True,
                              dirname=dirname)
            times[f'graph_{k}'].append(time.time())

            print(f"{bn} {k} gnn predict")
            if not files_exist_overwrite(overwrite,out_files[f'gnn_{k}']):
                predict(basename=bn,
                    analysis_type=k,
                    gpu_id=-1,
                    dirname=dirname)
            times[f'gnn_{k}'].append(time.time())

    print(f"{basename} ink detection")
    if not files_exist_overwrite(overwrite,out_files['ink']):
        detect_inks(basename=basename,
                compression=8.,
                ext=ext,
                dirname=dirname)
    times["ink"]=time.time()

    print(f"{basename} stitch")
    if not files_exist_overwrite(overwrite,out_files['ink']):
        if run_stitch_slide:
            stitch_slides(basename=basename,
                compression=4,
                ext=ext,
                dirname=dirname)
            times["stitch"]=time.time()

    return times


def run_series(patient="163_A1",
               input_dir="inputs",
               compression=1.,
               overwrite=True,
               record_time=False,
               ext=".npy",
               dirname=".",
               df_section_pieces_file="df_section_pieces.pkl",
               run_stitch_slide=True
               ):
    for f in glob.glob(os.path.join(input_dir,f"{patient}*{ext}")):
        basename=os.path.basename(f).replace(ext,"")#.replace(".tiff","").replace(".tif","").replace(".svs","")
        times=run_workflow_series(basename,
                            compression,
                            overwrite,
                            ext,
                            dirname,
                            df_section_pieces_file,
                            run_stitch_slide)
        if record_time:
            os.makedirs(os.path.join(dirname,"times"),exist_ok=True)
            pickle.dump(times,open(os.path.join(dirname,"times",f"{basename}.pkl"),'wb'))
