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
    print(command)
    result=os.popen(command).read()
    return result#f"Preprocessed {job_dict['basename']}"

def embed(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai cnn_predict --basename {job_dict['basename']} --analysis_type {job_dict['analysis_type']} --gpu_id -1"
    print(command)
    result=os.popen(command).read()
    return result#f"Embed CNN {job_dict['analysis_type']} {job_dict['basename']}"

def gnn_predict(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai gnn_predict --basename {job_dict['basename']} --analysis_type {job_dict['analysis_type']} --radius 256 --min_component_size 600 --gpu_id -1 --generate_graph True"
    print(command)
    result=os.popen(command).read()
    return result#f"GNN Predict {job_dict['analysis_type']} {job_dict['basename']}"

def gen_quality_scores(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai quality_score --basename {job_dict['basename']} "
    print(command)
    result=os.popen(command).read()
    return result#f"Quality {job_dict['basename']}"

def ink_detect(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai ink_detect --basename {job_dict['basename']} --compression 8 --ext {job_dict['ext']}"
    print(command)
    result=os.popen(command).read()
    return result#f"Ink {job_dict['basename']}"

def stitch_images(job, job_dict, memory="50G", cores=8, disk="1M"):
    command=f"cd {job_dict['job_dir']} && {job_dict['singularity_preamble']} arctic_ai ink_detect --basename {job_dict['basename']} --compression 8 --ext {job_dict['ext']}"
    print(command)
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
    quality_job=job.addChildJobFn(gen_quality_scores, job_dict, memory, cores, disk)
    for k in ['tumor','macro']:
        gnn_predict_jobs[k].addChild(quality_job)
    ink_job=job.addChildJobFn(ink_detect, job_dict, memory, cores, disk)
    jobs['preprocess'].addChild(ink_job)
    jobs['preprocess'].addChild(nuclei_job)
    
    return f"Processed {job_dict['basename']}"

def setup_deploy(job, job_dict, memory="2G", cores=2, disk="3G"):
    print(job_dict)
    os.chdir(job_dict['job_dir'])
    print(os.getcwd())
    jobs=[]
    print(os.path.join(job_dict['input_dir'],f"{job_dict['patient']}*{job_dict['ext']}"))
    print(glob.glob(os.path.join(job_dict['input_dir'],f"{job_dict['patient']}*{job_dict['ext']}")))
    for f in glob.glob(os.path.join(job_dict['input_dir'],f"{job_dict['patient']}*{job_dict['ext']}")):
        print(f)
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
               job_dir="/dartfs/rc/lab/V/VaickusL_slow/users/jlevy/arctic_ai/BCC_test_study",
               restart=False,
               logfile="",
               loglevel="",
               singularity_preamble="source ~/.bashrc && export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES && export SINGULARITYENV_PREPEND_PATH=/dartfs-hpc/rc/home/w/f003k8w/.local/bin/ && singularity  exec --nv -B $(pwd)  -B /scratch/ -B $(realpath ../../../..)  --bind ${HOME}:/mnt  /dartfs/rc/lab/V/VaickusL_slow/singularity_containers/PathFlow/pathflowgcn_new.img",
               slurm_args="--export=ALL --gres=gpu:1 --account=qdp-alpha --partition=v100_12 --nodes=1 --gpu_cmode=shared --ntasks-per-node=1 --time=1:00:00",#
               run_slurm=False,
               cores=2,
               memory="60G",
               disk="3G"):
    # to be run here: /dartfs/rc/lab/V/VaickusL_slow/users/jlevy/arctic_ai/BCC_test_study
    # toil clean toilWorkflowRun
    # python 7_toil_local_series_run.py
    # python 7_toil_local_series_run.py --run_slurm True
    # python 7_toil_local_series_run.py --restart
    # python 7_toil_local_series_run.py  --loglevel DEBUG  --patient 336_A1 # --logfile t.log
    # python 7_toil_local_series_run.py --patient 336_A1
    # for i in $(seq 2) ; do submit-job run-slurm-job -a "hostname" -a "echo GPU=\$CUDA_VISIBLE_DEVICES" -c "sleep 3h" -t 3 -acc qdp-alpha -p v100_12 --ppn 8 -t 12 -n 1 -ng 4 -gsm shared -mem 500 ; done
    # export CUDA_VISIBLE_DEVICES=$(($RANDOM % 4))
    # toil clean toilWorkflowRun &&  python 7_toil_local_series_run.py --patient 336_A1
    # TODO: Error out on bad os.system calls and pipe output
    # TODO: add TOIL SLURM environ input and singularity preamble
    # TODO: change memory reqs per job
    # TODO: Slurm executor
    # TODO: divide up even further?
    # TODO: DAG image export by writing connections to dag file and plotting
    # TODO: Throw error if output file not found
    # TODO: only using 1 gpu, need to split up, will fail on a few tasks, also launch more jobs at a time and asynchronous execution
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
        singularity_preamble="export CUDA_VISIBLE_DEVICES=$(($RANDOM % 4)) &&"+singularity_preamble
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
    print(rv)

if __name__=="__main__":
    fire.Fire(run_parallel)