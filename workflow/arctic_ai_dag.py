import datetime as dt
from airflow import DAG
from airflow.models import Variable
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from submit_hpc.job_runner import run_torque_job_
import glob, os
import numpy as np

TMP_SINGULARITY_EXPORTS="export SINGULARITYENV_CUDA_VISIBLE_DEVICES=\${gpuNum} && export SINGULARITYENV_PREPEND_PATH=/dartfs-hpc/rc/home/w/f003k8w/.local/bin/ && singularity exec --nv -B /dartfs/rc/lab/V/VaickusL_slow/  --bind ${HOME}:/mnt  /dartfs/rc/lab/V/VaickusL_slow/singularity_containers/PathFlow/pathflowgcn_new.img"
TMP_ADDITIONAL_OPTIONS="-A QDP-Alpha -l nodes=1:ppn=8:gpus=1 -l feature=v100"
TMP_TIME=1
TMP_NGPU=0
TMP_QUEUE="gpuq"
TMP_USER="f003k8w"
TMP_SLEEP=5

def compile_command(name,**kwargs):
    return f"{name} {' '.join(f'--{k} {v}' for k,v in kwargs.items())}"

def run_torque_job(command):
    command=f"{TMP_SINGULARITY_EXPORTS} arctic_ai {command}"
    run_torque_job_(command,
                    use_gpu=False,
                    additions=[],
                    queue=TMP_QUEUE,
                    time=TMP_TIME,
                    ngpu=TMP_NGPU,
                    additional_options=TMP_ADDITIONAL_OPTIONS,
                    self_gpu_avail=False,
                    imports=[],
                    monitor_job=True,
                    user=TMP_USER,
                    sleep=TMP_SLEEP,
                    verbose=True)

def compile_run_command(name, **kwargs):
    run_torque_job(compile_command(name, **kwargs))

def slide_done(basename):
    print(f"{basename} complete")

def create_dag(patient):
    default_args = {
        'owner': 'me',
        'start_date': dt.datetime(2017, 6, 1),
        'retries': 1,
        'retry_delay': dt.timedelta(minutes=15),
        'run_as_user': 'tmp_user'
    }
    dag= DAG(f'airflow_arctic_ai_patient_{patient}',
             default_args=default_args,
             )
    with dag:

        IDs=list(map(lambda f: os.path.basename(f).replace(".npy",""),glob.glob(f"inputs/{patient}*.npy")))

        tasks=dict()
        for i,ID in enumerate(IDs):
            tasks[f'{i}_preprocess'] = PythonOperator(task_id=f'{i}_preprocess',
                                         python_callable=compile_run_command,
                                         op_kwargs=dict(name="preprocess",
                                                        kwargs=dict(basename=ID,
                                                                    threshold=0.05,
                                                                    patch_size=256)))
            for analysis_type in ['tumor','macro']:
                tasks[f'{i}_cnn_predict_{analysis_type}'] = PythonOperator(task_id=f'{i}_cnn_predict_{analysis_type}',
                                             python_callable=compile_run_command,
                                             op_kwargs=dict(name="cnn_predict",
                                                            kwargs=dict(basename=ID,
                                                                        analysis_type=analysis_type,
                                                                        gpu_id=-1)))
                tasks[f'{i}_gnn_predict_{analysis_type}'] = PythonOperator(task_id=f'{i}_gnn_predict_{analysis_type}',
                                             python_callable=compile_run_command,
                                             op_kwargs=dict(name="gnn_predict",
                                                            kwargs=dict(basename=ID,
                                                                        analysis_type=analysis_type,
                                                                        radius=256,
                                                                        min_component_size=600,
                                                                        gpu_id=-1)))
            tasks[f'{i}_nuclei_predict'] = PythonOperator(task_id=f'{i}_nuclei_predict',
                                         python_callable=compile_run_command,
                                         op_kwargs=dict(name="nuclei_predict",
                                                        kwargs=dict(basename=ID,
                                                                    gpu_id=-1)))
            tasks[f'{i}_quality_score'] = PythonOperator(task_id=f'{i}_quality_score',
                                         python_callable=compile_run_command,
                                         op_kwargs=dict(name="quality_score",
                                                        kwargs=dict(basename=ID)))
            tasks[f'{i}_ink_detect'] = PythonOperator(task_id=f'{i}_ink_detect',
                                         python_callable=compile_run_command,
                                         op_kwargs=dict(name="ink_detect",
                                                        kwargs=dict(basename=ID,
                                                                    compression=8)))
            tasks[f'{i}_slide_done'] = PythonOperator(task_id=f'{i}_slide_done',
                                         python_callable=slide_done,
                                         op_kwargs=dict(basename=ID))
        tasks[f'dump_results'] = PythonOperator(task_id='dump_results',
                                         python_callable=compile_run_command,
                                         op_kwargs=dict(name="dump_results",
                                                        kwargs=dict(patient=patient,
                                                                scheme="2/1"))) # fix scheme

        for i,ID in IDs:
            tasks[f'{i}_preprocess'] >> [tasks[f'{i}_cnn_predict_tumor'],tasks[f'{i}_cnn_predict_macro'],tasks[f'{i}_nuclei_predict'],tasks[f'{i}_ink_detect']]
            for analysis_type in ['macro','tumor']:
                tasks[f'{i}_cnn_predict_{analysis_type}']>>tasks[f'{i}_gnn_predict_{analysis_type}']
            [tasks[f'{i}_gnn_predict_{analysis_type}'] for analysis_type in ['tumor','macro']]>> tasks[f'{i}_quality_score']

            tasks['quality_score'] << [tasks['cnn_predict'],tasks['gnn_predict']]
            [tasks['nuclei_predict'],tasks['ink_detect'],tasks['quality_score']] >> tasks[f'{i}_slide_done']
        [tasks[f'{i}_slide_done'] for i in range(len(IDs))] >> tasks['dump_results']
    return dag

# patient = Variable.get('patient', default_var='')
patients=np.unique(np.vectorize(lambda x: os.path.basename(x)[:6])(glob.glob("inputs/*.npy")))
for patient in patients:
    globals()[patient] = create_dag(patient)
    # READ DATABASE
    # ADD DZI Steps
