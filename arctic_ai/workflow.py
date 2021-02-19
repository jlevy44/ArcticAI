import glob, os
from .preprocessing import preprocess
from .cnn_prediction import generate_embeddings
from .generate_graph import create_graph_data
from .gnn_prediction import predict
from .quality_scores import generate_quality_scores
from .ink_detection import detect_inks
from .compile_results import dump_results
from .dzi_writer import npy2dzi
from .case_prototype import Case

def run_workflow_series(basename, compression):
    print(f"{basename} preprocessing")
    preprocess(basename=basename,
               threshold=0.05,
               patch_size=256)

    for k in ['tumor','macro']:
        print(f"{basename} {k} embedding")
        generate_embeddings(basename=basename,
                            analysis_type=k,
                           gpu_id=0)

        print(f"{basename} {k} build graph")
        create_graph_data(basename=basename,
                          analysis_type=k,
                          radius=256,
                          min_component_size=600)

        print(f"{basename} {k} gnn predict")
        predict(basename=basename,
                analysis_type=k,
                gpu_id=0)

    print(f"{basename} quality assessment")
    generate_quality_scores(basename)

    print(f"{basename} ink detection")
    detect_inks(basename=basename,
                compression=8)

    print(f"{basename} nuclei detection")
    predict_nuclei(basename=basename,
                   gpu_id=0)


def run_series(patient="163_A1",
               input_dir="inputs",
               scheme="2/1",
               compression=1.):
    for f in glob.glob(os.path.join(input_dir,f"{patient}*.npy")):
        run_workflow_series(os.path.basename(f).replace(".npy",""),
                            compression)

    dump_results(patient,scheme)
    case=Case(patient=patient)
    for k in ['image','nuclei','tumor','ink']:
        case.extract2dzi(k)
