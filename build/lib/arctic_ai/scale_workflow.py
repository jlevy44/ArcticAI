"""Parallel Workflow 
==========
Contains functions for parallel processing of tissue sections."""
from toil.job import Job
import subprocess, os
import glob
import fire

def files_exist_overwrite(overwrite, files):
    return (not overwrite) and all([os.path.exists(file) for file in files])

def generate_output_file_names(basename):
    out_files=dict()
    out_files['preprocess']=[f"masks/{basename}_{k}_map.npy" for k in ['tumor','macro']]+[f"patches/{basename}_{k}_map.npy" for k in ['tumor','macro']]
    for k in ['macro','tumor']:
        out_files[f'cnn_{k}']=[f"cnn_embeddings/{basename}_{k}_map.pkl"]
        out_files[f'gnn_{k}']=[f"gnn_results/{basename}_{k}_map.pkl",f"graph_datasets/{basename}_{k}_map.pkl"]
    out_files['quality']=[f"quality_scores/{basename}.pkl"]
    out_files['ink']=[f"detected_inks/{basename}_thumbnail.npy"]
    out_files['nuclei']=[f"nuclei_results/{basename}.npy"]
    return out_files

def preprocess(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai preprocess --basename {job_dict['basename']} --threshold 0.05 --patch_size 256 --ext {job_dict['ext']} --compression {job_dict['compression']}"
    # print(command)
    result=os.popen(command).read()
    return result#f"Preprocessed {job_dict['basename']}"

def embed(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai cnn_predict --basename {job_dict['basename']} --analysis_type {job_dict['analysis_type']} --gpu_id -1"
    # print(command)
    result=os.popen(command).read()
    return result#f"Embed CNN {job_dict['analysis_type']} {job_dict['basename']}"

def gnn_predict(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai gnn_predict --basename {job_dict['basename']} --analysis_type {job_dict['analysis_type']} --radius 256 --min_component_size 600 --gpu_id -1 --generate_graph True"
    # print(command)
    result=os.popen(command).read()
    return result#f"GNN Predict {job_dict['analysis_type']} {job_dict['basename']}"

def gen_quality_scores(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai quality_score --basename {job_dict['basename']} "
    # print(command)
    result=os.popen(command).read()
    return result#f"Quality {job_dict['basename']}"

def ink_detect(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai ink_detect --basename {job_dict['basename']} --compression 8 --ext {job_dict['ext']}"
    # print(command)
    result=os.popen(command).read()
    return result#f"Ink {job_dict['basename']}"

def stitch_images(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai ink_detect --basename {job_dict['basename']} --compression 8 --ext {job_dict['ext']}"
    # print(command)
    result=os.popen(command).read()
    return result#f"Ink {job_dict['basename']}"

def deploy_patient(job, job_dict, memory="2G", cores=2, disk="1M"):
    os.chdir(job_dict['job_dir'])
    out_files=generate_output_file_names(job_dict['basename'])
    
    jobs={}
    preprocess_job=job.addChildJobFn(preprocess, job_dict, memory, cores, disk)
    jobs['preprocess']=preprocess_job
    
    embed_jobs={}
    gnn_predict_jobs={}
    for k in ['tumor','macro']:
        job_dict_k=job_dict.copy()
        job_dict_k['analysis_type']=k
        embed_jobs[k]=job.addChildJobFn(embed, job_dict_k, memory, cores, disk)
        gnn_predict_jobs[k]=job.addChildJobFn(gnn_predict, job_dict_k, memory, cores, disk)
        jobs['preprocess'].addChild(embed_jobs[k])
        embed_jobs[k].addChild(gnn_predict_jobs[k])
    jobs['embed']=embed_jobs
    jobs['gnn']=gnn_predict_jobs
    # quality_job=job.addChildJobFn(gen_quality_scores, job_dict, memory, cores, disk)
    # for k in ['tumor','macro']:
    #     gnn_predict_jobs[k].addChild(quality_job)
    ink_job=job.addChildJobFn(ink_detect, job_dict, memory, cores, disk)
    jobs['preprocess'].addChild(ink_job)
    jobs['preprocess'].addChild(nuclei_job)
    
    return f"Processed {job_dict['basename']}"

def setup_deploy(job, job_dict, memory="2G", cores=2, disk="3G"):
    os.chdir(job_dict['job_dir'])
    jobs=[]
    for f in glob.glob(os.path.join(job_dict['input_dir'],f"{job_dict['patient']}*{job_dict['ext']}")):
        # print(f)
        basename=os.path.basename(f).replace(job_dict['ext'],"")
        job_dict_f=dict(basename=basename, 
                        compression=job_dict['compression'], 
                        overwrite=job_dict['overwrite'], 
                        ext=job_dict['ext'],
                        job_dir=job_dict['job_dir'],
                        singularity_preamble=job_dict['singularity_preamble'])
        patient_job=job.addChildJobFn(deploy_patient, job_dict_f, memory, cores, disk)
        jobs.append(patient_job)
    return [patient_job.rv() for patient_job in jobs]
                    

def run_parallel(patient="",
               input_dir="inputs",
               scheme="2/1",
               compression=6.,
               overwrite=True,
               record_time=False,
               extract_dzi=False,
               ext=".tif",
               job_dir="./",
               restart=False,
               logfile="",
               loglevel="",
               cuda_visible_devices="$CUDA_VISIBLE_DEVICES",
               singularity_img_path="arcticai.img",
               run_slurm=False,
               cores=2,
               memory="60G",
               disk="3G",
               cuda_device="$(($RANDOM % 4))",
               prepend_path="$(realpath ~)/.local/bin/",
               slurm_gpus=1,
               slurm_account="qdp-alpha",
               slurm_partition="v100_12",
               time=1,
               gpu_cmode="exclusive"):
    """
    Runs the image processing workflow in parallel across multiple tissue sections simultaneously.

    Parameters:
    -----------
    patient : str, optional
        The patient identifier. Default is an empty string.
    input_dir : str, optional
        The directory containing the input images. Default is 'inputs'.
    scheme : str, optional
        The grid scheme used for image processing. Default is '2/1'.
    compression : float, optional
        The compression level for the output images. Default is 6.0.
    overwrite : bool, optional
        Whether or not to overwrite existing output files. Default is True.
    record_time : bool, optional
        Whether or not to record the processing time for each image. Default is False.
    extract_dzi : bool, optional
        Whether or not to extract deep zoom images. Default is False.
    ext : str, optional
        The file extension for input images. Default is '.tif'.
    job_dir : str, optional
        The directory for job management. Default is './'.
    restart : bool, optional
        Whether or not to restart a previous job. Default is False.
    logfile : str, optional
        The path to the log file. Default is an empty string.
    loglevel : str, optional
        The level of verbosity for logging. Default is an empty string.
    cuda_visible_devices : str, optional
        The value for the CUDA_VISIBLE_DEVICES environment variable. Default is '$CUDA_VISIBLE_DEVICES'.
    singularity_img_path : str, optional
        The path to the Singularity image file. Default is 'arcticai.img'.
    run_slurm : bool, optional
        Whether or not to run the job using the Slurm scheduler. Default is False.
    cores : int, optional
        The number of CPU cores to use. Default is 2.
    memory : str, optional
        The amount of memory to allocate for each job. Default is '60G'.
    disk : str, optional
        The amount of disk space to allocate for each job. Default is '3G'.
    cuda_device : str, optional
        The value for the CUDA device ID. Default is '$(($RANDOM % 4))'.
    prepend_path : str, optional
        The value for the PREPEND_PATH environment variable. Default is '$(realpath ~)/.local/bin/'.
    slurm_gpus : int, optional
        The number of GPUs to allocate when running the job on a Slurm cluster. Default is 1.
    slurm_account : str, optional
        The name of the Slurm account to use. Default is 'qdp-alpha'.
    slurm_partition : str, optional
        The name of the Slurm partition to use. Default is 'v100_12'.
    time : int, optional
        The maximum amount of time to allocate for the job, in hours. Default is 1.
    gpu_cmode : str, optional
        Slurm option that controls GPU sharing mode between multiple users. Default is exclusive though can be toggled to shared
    """
    singularity_preamble=f"source ~/.bashrc && export SINGULARITYENV_CUDA_VISIBLE_DEVICES={cuda_visible_devices} && export SINGULARITYENV_PREPEND_PATH={prepend_path} && singularity  exec --nv -B $(pwd)  -B /scratch/  --bind ${HOME}:/mnt  {singularity_img_path}"
    slurm_args=f"--export=ALL --gres=gpu:{slurm_gpus} --account={slurm_account} --partition={slurm_partition} --nodes=1 --ntasks-per-node=1 --time={time}:00:00 "
    if gpu_cmode=="shared":
        slurm_args+="--gpu_cmode=shared"
    options = Job.Runner.getDefaultOptions("./toilWorkflowRun")
    options.restart=restart
    options.defaultCores=cores
    options.defaultMemory=memory
    options.defaultDisk=disk
    options.clean = "always"
    if run_slurm: 
        os.environ["TOIL_SLURM_ARGS"]=slurm_args
        options.batchSystem="slurm"
        options.disableCaching=True
        options.statePollingWait = 5
        options.maxLocalJobs = 100
        options.targetTime = 1
    else:
        singularity_preamble=f"export CUDA_VISIBLE_DEVICES={cuda_device} &&"+singularity_preamble
    if loglevel: options.logLevel=loglevel
    if logfile: options.logFile=logfile
    job_dict=dict(patient=patient,
               input_dir=input_dir,
               scheme=scheme,
               compression=compression,
               overwrite=overwrite,
               record_time=record_time,
               extract_dzi=extract_dzi,
               ext=ext,
               job_dir=job_dir,
               singularity_preamble=singularity_preamble)
    j = Job.wrapJobFn(setup_deploy, job_dict)
    rv = Job.Runner.startToil(j, options)

# if __name__=="__main__":
#     fire.Fire(run_parallel)