import glob, os, time, pickle
from .preprocessing import preprocess_old
from .cnn_prediction import generate_embeddings_old
from .generate_graph import create_graph_data
from .gnn_prediction import predict
from .quality_scores import generate_quality_scores
from .ink_detection import detect_inks_old
from .compile_results import dump_results
from .nuclei_prediction import predict_nuclei
from .image_stitch import npy2dzi
from .case_prototype import Case
import warnings
from functools import partial



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

def run_workflow_series(basename, compression, overwrite, ext):
    print(f"{basename} preprocessing")

    out_files=generate_output_file_names(basename)

    times=dict()
    times['start']=time.time()

    img_shape=None
    if not files_exist_overwrite(overwrite,out_files['preprocess']):
        img_shape=preprocess_old(basename=basename,
               threshold=0.05,
               patch_size=256,
               ext=ext)

    times['preprocess']=time.time()

    for k in ['tumor','macro']:
        print(f"{basename} {k} embedding")
        if not files_exist_overwrite(overwrite,out_files[f'cnn_{k}']):
            generate_embeddings_old(basename=basename,
                            analysis_type=k,
                           gpu_id=-1)
        times[f'cnn_{k}']=time.time()

        print(f"{basename} {k} build graph")
        if not files_exist_overwrite(overwrite,out_files[f'graph_data_{k}']):
            create_graph_data(basename=basename,
                          analysis_type=k,
                          radius=256,
                          min_component_size=600)
        times[f'graph_{k}']=time.time()

        print(f"{basename} {k} gnn predict")
        if not files_exist_overwrite(overwrite,out_files[f'gnn_{k}']):
            predict(basename=basename,
                analysis_type=k,
                gpu_id=-1)
        times[f'gnn_{k}']=time.time()

    print(f"{basename} quality assessment")
    if not files_exist_overwrite(overwrite,out_files['quality']):
        generate_quality_scores(basename)
    times["quality"]=time.time()

    print(f"{basename} ink detection")
    if not files_exist_overwrite(overwrite,out_files['ink']):
        detect_inks_old(basename=basename,
                compression=8,
                ext=ext)
    times["ink"]=time.time()

    print(f"{basename} nuclei detection")
    if not files_exist_overwrite(overwrite,out_files['nuclei']):
        predict_nuclei(basename=basename,
                   gpu_id=-1,
                   ext=ext,
                   img_shape=img_shape)
    times["nuclei"]=time.time()
    return times


def run_series(patient="163_A1",
               input_dir="inputs",
               scheme="2/1",
               compression=1.,
               overwrite=True,
               record_time=False,
               extract_dzi=False,
               ext=".npy"
               ):
    warnings.warn(
            "Will replace with new loop to process sections separately",
            DeprecationWarning
        )
    times=dict()
    for f in glob.glob(os.path.join(input_dir,f"{patient}*{ext}")):
        basename=os.path.basename(f).replace(ext,"")#.replace(".tiff","").replace(".tif","").replace(".svs","")
        times[basename]=run_workflow_series(basename,
                            compression,
                            overwrite,
                            ext)
    if record_time:
        os.makedirs("times",exist_ok=True)
        pickle.dump(times,open(os.path.join("times",f"{patient}.pkl"),'wb'))
    dump_results(patient,scheme)
    if extract_dzi:
        case=Case(patient=patient)
        for k in ['image','nuclei','tumor','ink']:
            case.extract2dzi(k)
