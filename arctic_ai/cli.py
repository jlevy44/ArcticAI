"""
CLI 
==========

Contains functions for creating command line interfaces.
"""
import fire, os

class Commands(object):
    """A class for defining command line interface methods using Google Fire.

    Attributes:
        None
    """
    def __init__(self):
        pass

    def preprocess(self, 
                basename: str = "163_A1a", 
                threshold: float = 0.05, 
                patch_size: int = 256, 
                ext: str = ".npy", 
                secondary_patch_size: int = 0, 
                df_section_pieces_file: str = "", 
                image_mask_compression: float = 8., 
                dirname: str = ".") -> None:
        """
        Preprocesses an image and generates patches for training or testing.

        Args:
            basename: The base filename of the image to be preprocessed.
            threshold: The maximum fraction of a patch that can be blank. Default is 0.05.
            patch_size: The size of the patches to be generated. Default is 256.
            ext: The file extension of the input image. Default is ".npy".
            secondary_patch_size: The size of the patches for secondary processing. Default is 0.
            df_section_pieces_file: The filename of the file containing metadata about image patches. Default is "section_pieces.pkl".
            image_mask_compression: The degree of compression applied to the image mask. Default is 8.
            dirname: The directory where input and output files are stored. Default is ".".
            
        Returns:
            None
        """
        from arctic_ai.preprocessing import preprocess
        preprocess(basename,threshold,patch_size,ext,secondary_patch_size,df_section_pieces_file=df_section_pieces_file,image_mask_compression=image_mask_compression, dirname=dirname)

    def cnn_predict(self,
                basename: str = "163_A1a",
                analysis_type: str = "tumor",
                gpu_id: int = -1) -> None:
        """
        Generates embeddings from preprocessed images using a trained CNN model.
        
        Parameters
        ----------
        basename : str, optional
            The base filename of the preprocessed image, default is "163_A1a".
        analysis_type : str, optional
            The type of analysis to be performed, default is "tumor".
        gpu_id : int, optional
            The ID of the GPU to be used for prediction, default is -1 (CPU).
        """
        from arctic_ai.cnn_prediction import generate_embeddings
        generate_embeddings(basename,analysis_type,gpu_id)

    def graph_creation(self,
                   basename: str = "163_A1a",
                   analysis_type: str = "tumor",
                   radius: int = 256,
                   min_component_size: int = 600,
                   no_component_break: bool = True) -> None:
        """
        Generates a graph based on embeddings of image patches.

        Parameters
        ----------
        basename : str, optional
            The base filename of the image to be used for generating the graph. Default is "163_A1a".
        analysis_type : str, optional
            The type of analysis to be performed on the image. Default is "tumor".
        radius : int, optional
            The radius used for generating the graph. Default is 256.
        min_component_size : int, optional
            The minimum size for a connected component in the graph. Default is 600.
        no_component_break : bool, optional
            If True, the function will not break up large connected components. Default is True.

        Returns
        -------
        None
        """
        from arctic_ai.generate_graph import create_graph_data
        create_graph_data(basename,analysis_type,radius,min_component_size)

    def gnn_predict(self,
                      basename="163_A1a",
                      analysis_type="tumor",
                      radius=256,
                      min_component_size=600,
                      gpu_id=-1,
                      generate_graph=True,
                      no_component_break=True):
        """Run GNN prediction on patches.

        Parameters
        ----------
        basename : str, optional
            The base filename of the image to be segmented. Default is "163_A1a".
        analysis_type : str, optional
            The type of analysis to perform. Default is "tumor".
        radius : int, optional
            The radius used to generate the graphs. Default is 256.
        min_component_size : int, optional
            The minimum size of the connected components in graph. Default is 600.
        gpu_id : int, optional
            The ID of the GPU to use for prediction. Default is -1, which uses the CPU.
        generate_graph : bool, optional
            Whether to generate the graph data from the preprocessed image. Default is True.
        no_component_break : bool, optional
            Whether to avoid breaking connected components across patches. Default is True.

        Returns
        -------
        None
        """
        from arctic_ai.gnn_prediction import predict
        if generate_graph:
            from arctic_ai.generate_graph import create_graph_data
            create_graph_data(basename,analysis_type,radius,min_component_size,no_component_break=no_component_break)
        predict(basename,analysis_type,gpu_id)

    def nuclei_predict(self, 
                       predictor_dir="./", 
                       predictor_file="", 
                       patch_file="", 
                       threshold=0.05, 
                       savenpy='pred.npy') -> None:
        '''
        The nuclei_predict function takes in a set of arguments including predictor_dir, predictor_file, patch_file, threshold, and savenpy. It loads a model from the specified predictor_dir and predictor_file, runs it on a set of patches in patch_file, and saves the output predictions as an npy stack and label dictionary if savenpy is not None. If savexml is not None, it also saves predictions to an ASAP xml.
        The function loads the model, flips the patches, runs the model on the patches, and saves the predictions in the specified formats if requested.
        Args:
            predictor_dir (str): path to model folder
            predictor_file (str): filename of model in the folder (don't include path to folder)
            patch_file (str): path to an npy stack of patches
            classifier_type (BasePredictor): class (from predict.py)
            panoptic (bool): whether the model performs panoptic segmentation. If false, it is assumed to do instance segmentation.
            n (int): number of classes the model classifies into 
            threshold (float): threshold to use for model 
            savenpy (str): Path to file to which to save npy output. The output is a numpy stack of the the masks for each patch. In the mask, nuclei are given a non-zero integer, the ID of the the instance it is a part of. A pickled dictionary mapping the instance ID to the class label is also outputted. If savenpy=None, predictions are not saved in an npy format. 
            savexml (str): Path to file to which to save xml output (ASAP format). If savexml=None, predictions are not saved in an xml format. 
            patch_coords (str): an extra pkl file which specifies the x,y (x is row, y is col) metadata for the patches. Must be provided if exporting to xml, since location is a part of the ASAP format
        '''
        # raise NotImplementedError("Removed temporarily, adding back soon")
        from arctic_ai.detection_workflows.nuclei_cli import detect_from_patches
        detect_from_patches(predictor_dir=predictor_dir, 
                       predictor_file=predictor_file, 
                       patch_file=patch_file, 
                       threshold=threshold, 
                       savenpy=savenpy)

    def quality_score(self,
                      basename="163_A1a"):
        from arctic_ai.quality_scores import generate_quality_scores
        generate_quality_scores(basename)

    def ink_detect(self,
                   basename="163_A1a",
                   compression=8):
        """
        Detect inks in the specified image.

        Parameters
        ----------
        basename : str, optional
            The base name of the image to detect inks in, by default "163_A1a"
        compression : float, optional
            The compression factor to use when detecting inks, by default 8.

        Returns
        -------
        None
            The function saves the detected inks to a pickle file, but does not return anything.
        """
        from arctic_ai.ink_detection import detect_inks
        detect_inks(basename,compression)

    def dump_results(self,
                     patient="163_A1",
                     scheme="2/1"):
        raise NotImplementedError("Deprecated")
        from arctic_ai.compile_results import dump_results
        dump_results(patient,scheme)

    def run_series_old(self,
                   patient="163_A1",
                   input_dir="inputs",
                   scheme="2/1",
                   compression=1.,
                   overwrite=True,
                   record_time=False,
                   extract_dzi=False,
                   ext=".npy"):
        raise NotImplementedError("Deprecated")
        from arctic_ai.workflow import run_series
        run_series(patient,input_dir,scheme,compression,overwrite,record_time,extract_dzi,ext)

    def run_series(self,
                   patient="163_A1",
                   input_dir="inputs",
                   compression=1.,
                   overwrite=True,
                   record_time=False,
                   ext=".npy",
                   dirname=".",
                   df_section_pieces_file="df_section_pieces.pkl",
                   run_stitch_slide=True):
        """Runs the entire image analysis workflow on a given patient.

        Parameters
        ----------
        patient : str, optional
            The patient ID to be processed. Default is "163_A1".
        input_dir : str, optional
            The directory where the input files are stored. Default is "inputs".
        compression : float, optional
            The degree of compression applied to the input image. Default is 1.0.
        overwrite : bool, optional
            If True, overwrite existing output files. Default is True.
        record_time : bool, optional
            If True, record the time taken for each step of the workflow. Default is False.
        ext : str, optional
            The file extension of the input image. Default is ".npy".
        dirname : str, optional
            The directory where input and output files are stored. Default is ".".
        df_section_pieces_file : str, optional
            The filename of the file containing metadata about image patches. Default is "df_section_pieces.pkl".
        This method runs the entire image analysis workflow on a given patient, with options to customize the input and output directories, degree of image compression, file extension, and whether to overwrite existing output files. If record_time is set to True, the time taken for each step of the workflow will be recorded. The default values for patient, input_dir, compression, overwrite, ext, dirname, and df_section_pieces_file are "163_A1", "inputs", 1.0, True, ".npy", ".", and "df_section_pieces.pkl", respectively.
        """
        from arctic_ai.workflow import run_series
        run_series(patient,input_dir,compression,overwrite,record_time,ext,dirname,df_section_pieces_file,run_stitch_slide)

    def run_parallel(self,
                    patient="163_A1",
                    input_dir="inputs",
                    compression=1.,
                    overwrite=True,
                    record_time=False,
                    ext=".npy",
                    dirname=".",
                    df_section_pieces_file="df_section_pieces.pkl",
                    run_stitch_slide=True):
            """Runs the entire image analysis workflow on a given patient, in parallel using either a local executor or slurm (more executors will be added).

            Parameters
            ----------
            patient : str, optional
                The patient ID to be processed. Default is "163_A1".
            input_dir : str, optional
                The directory where the input files are stored. Default is "inputs".
            compression : float, optional
                The degree of compression applied to the input image. Default is 1.0.
            overwrite : bool, optional
                If True, overwrite existing output files. Default is True.
            record_time : bool, optional
                If True, record the time taken for each step of the workflow. Default is False.
            ext : str, optional
                The file extension of the input image. Default is ".npy".
            dirname : str, optional
                The directory where input and output files are stored. Default is ".".
            df_section_pieces_file : str, optional
                The filename of the file containing metadata about image patches. Default is "df_section_pieces.pkl".
            This method runs the entire image analysis workflow on a given patient, with options to customize the input and output directories, degree of image compression, file extension, and whether to overwrite existing output files. If record_time is set to True, the time taken for each step of the workflow will be recorded. The default values for patient, input_dir, compression, overwrite, ext, dirname, and df_section_pieces_file are "163_A1", "inputs", 1.0, True, ".npy", ".", and "df_section_pieces.pkl", respectively.
            """
            from arctic_ai.scale_workflow import run_parallel
            run_parallel(patient,input_dir,compression,overwrite,record_time,ext,dirname,df_section_pieces_file,run_stitch_slide)

    def tif2npy(self,
                in_file='',
                out_dir='./'):
        """Converts a .tif file to a .npy file.

        Parameters
        ----------
        in_file : str
            The path to the input .tif file.
        out_dir : str
            The path to the output directory where the .npy file will be saved.
        """
        from arctic_ai.utils import tif2npy
        tif2npy(in_file,out_dir)

    def dzi_folder_setup(self):
        from arctic_ai.utils import return_osd_template
        return_osd_template()

    def im2dzi(self,
               in_file='',
               out_dir='./',
               compression: float = 1.):
        """
        Converts an input image file to Deep Zoom Image format.

        Parameters
        ----------
        in_file : str
            The path to the input image file.
        out_dir : str
            The output directory where the DZI file will be written.
            Default is "./".
        compression : float
            The degree of compression applied to the DZI file. Default is 1.

        Returns
        -------
        None
        """
        from arctic_ai.image_stitch import npy2dzi
        basename,_=os.path.splitext(os.path.basename(in_file))
        npy2dzi(npy_file=in_file,
                    dzi_out=os.path.join(out_dir,f"{basename}.dzi"),
                    compression=compression)

    def write_dzis(self,
                   basename="",
                    compression=4,
                    dirname=".",
                    ext=".tif"):
        raise NotImplementedError("Deprecated")
        from arctic_ai.image_stitch import stitch_slides
        stitch_slides(basename=basename,
                     compression=compression,
                     dirname=dirname,
                     ext=ext)


    def extract_dzis(self,
                     patient='163_A1',
                     overwrite_scheme='',
                     types=['image','tumor','macro']):
        raise NotImplementedError("Deprecated")
        from arctic_ai.case_prototype import Case
        case=Case(patient=patient,overwrite_scheme=overwrite_scheme)
        for k in types:
            case.extract2dzi(k)

def main():
    fire.Fire(Commands)

if __name__=="__main__":
    main()
