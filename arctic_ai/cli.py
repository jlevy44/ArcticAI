import fire

class Commands(object):
    def __init__(self):
        pass

    def preprocess(self,
                   basename="163_A1a",
                   threshold=0.05,
                   patch_size=256):
        from arctic_ai.preprocessing import preprocess
        preprocess(basename,threshold,patch_size)

    def cnn_predict(self,
                    basename="163_A1a",
                    analysis_type="tumor",
                    gpu_id=-1):
        from arctic_ai.cnn_prediction import generate_embeddings
        generate_embeddings(basename,analysis_type,gpu_id)

    def gnn_predict(self,
                      basename="163_A1a",
                      analysis_type="tumor",
                      radius=256,
                      min_component_size=600,
                      gpu_id=-1):
        from arctic_ai.generate_graph import create_graph_data
        from arctic_ai.gnn_prediction import predict
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
                   overwrite=True):
        from arctic_ai.workflow import run_series
        run_series(patient,input_dir,scheme,compression,overwrite)

def main():
    fire.Fire(Commands)

if __name__=="__main__":
    main()
