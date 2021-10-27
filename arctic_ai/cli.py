import fire

class Commands(object):
    def __init__(self):
        pass

    def preprocess(self,
                   basename="163_A1a",
                   threshold=0.05,
                   patch_size=256,
                   ext=".npy"):
        from arctic_ai.preprocessing import preprocess
        preprocess(basename,threshold,patch_size,ext)

    def cnn_predict(self,
                    basename="163_A1a",
                    analysis_type="tumor",
                    gpu_id=-1):
        from arctic_ai.cnn_prediction import generate_embeddings
        generate_embeddings(basename,analysis_type,gpu_id)

    def graph_creation(self,
                      basename="163_A1a",
                      analysis_type="tumor",
                      radius=256,
                      min_component_size=600):
        from arctic_ai.generate_graph import create_graph_data
        create_graph_data(basename,analysis_type,radius,min_component_size)

    def gnn_predict(self,
                      basename="163_A1a",
                      analysis_type="tumor",
                      radius=256,
                      min_component_size=600,
                      gpu_id=-1,
                      generate_graph=True):
        from arctic_ai.gnn_prediction import predict
        if generate_graph:
            from arctic_ai.generate_graph import create_graph_data
            create_graph_data(basename,analysis_type,radius,min_component_size)
        predict(basename,analysis_type,gpu_id)

    def nuclei_predict(self,
                       basename="163_A1a",
                       gpu_id=-1):
        from arctic_ai.nuclei_prediction import predict_nuclei
        predict_nuclei(basename,gpu_id)

    def quality_score(self,
                      basename="163_A1a"):
        from arctic_ai.quality_scores import generate_quality_scores
        generate_quality_scores(basename)

    def ink_detect(self,
                   basename="163_A1a",
                   compression=8):
        from arctic_ai.ink_detection import detect_inks
        detect_inks(basename,compression)

    def dump_results(self,
                     patient="163_A1",
                     scheme="2/1"):
        from arctic_ai.compile_results import dump_results
        dump_results(patient,scheme)

    def run_series(self,
                   patient="163_A1",
                   input_dir="inputs",
                   scheme="2/1",
                   compression=1.,
                   overwrite=True,
                   record_time=False,
                   extract_dzi=False,
                   ext=".npy"):
        from arctic_ai.workflow import run_series
        run_series(patient,input_dir,scheme,compression,overwrite,record_time,extract_dzi,ext)

    def tif2npy(self,
                in_file='',
                out_dir='./'):
        from arctic_ai.utils import tif2npy
        tif2npy(in_file,out_dir)

    def extract_dzis(self,
                     patient='163_A1',
                     overwrite_scheme='',
                     types=['image','tumor','macro']):
        from arctic_ai.case_prototype import Case
        case=Case(patient=patient,overwrite_scheme=overwrite_scheme)
        for k in types:
            case.extract2dzi(k)

def main():
    fire.Fire(Commands)

if __name__=="__main__":
    main()
