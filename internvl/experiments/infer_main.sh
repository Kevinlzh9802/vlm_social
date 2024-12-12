#!/bin/bash

# Usage message
usage() {
    echo "Usage: $0 --model_name <model_name> --visual_strat <visual_strat> --task <task>"
    echo "  model_name: internvl1b, internvl2b, internvl4b"
    echo "  visual_strat: multi, concat"
    echo "  task: fform, cgroup"
    exit 1
}

# Default values
model_name=""
visual_strat=""
task=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) model_name="$2"; shift ;;
        --visual_strat) visual_strat="$2"; shift ;;
        --task) task="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

# Call the Python script with the validated arguments
python ../evaluate.py --model_name "$model_name" --visual_strat "$visual_strat" --task "$task"
#apptainer exec --nv --bind /tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan:/mnt/zonghuan --bind /home/nfs/zli33:/mnt/zli33 /tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_builds/containers/internvl2-8.sif python /home/nfs/zli33/projects/vlm_social/internvl/localize_person.py --model_name "$model_name" --visual_strat "$visual_strat" --task "$task"

# sample usage
# ./infer_main.sh --model_name internvl2b --visual_strat concat --task cgroup
