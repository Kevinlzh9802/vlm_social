#!/bin/sh
#SBATCH --partition=insy,general # Request partition. Default is 'general'
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=10:00:00      # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1          # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2   # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=32768          # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes.
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_infer/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_infer/slurm_%j.err # Set name of error log. %j is the Slurm jobId

#SBATCH --gres=gpu:a40:1 # Request 1 GPU

# Measure GPU usage of your job (initialization)
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')

/usr/bin/nvidia-smi # Check sbatch settings are working (it should show the GPU that you requested)

# Remaining job commands go below here. For example, to run python code that makes use of GPU resources:
model_name=""
visual_strat=""
task=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --visual_strat) visual_strat="$2"; shift ;;
        --task) task="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done
# Uncomment these lines and adapt them to load the software that your job requires
# module use /opt/insy/modulefiles          # Use DAIC INSY software collection
# module load cuda/12.4 cudnn/12-8.9.1.23 # Load certain versions of cuda and cudnn
# srun bash infer_main.sh --model_name InternVL2_5-1B --visual_strat gallery --task locate # Computations should be started with 'srun'. For example:
srun bash infer_main.sh --model_name "$model_name" --visual_strat "$visual_strat" --task "$task" # Computations should be started with 'srun'. For example:
# Measure GPU usage of your job (result)
/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"


