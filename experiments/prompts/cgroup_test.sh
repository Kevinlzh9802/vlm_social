#!/bin/sh

prompting () {
    # shellcheck disable=SC3043
    local model_exec="$1"
    local model="$2"
    local mmproj="$3"
    local image="$4"
    local prompt="$5"
    local result_file="$6"
    local model_log="$7"

    local output="$($model_exec -m "$model" --mmproj "$mmproj" --image "$image" -c 4096 -p "$prompt")"
    # # echo "-----------------------------------------------------------------------------"

    timestamp=$(date +%Y/%m/%d,%H:%M:%S)
    {
      echo "$timestamp";
      echo "$image";
      echo "$output";
      } >> "$result_file"
    echo "----------------------------------------"
}

if [[ "$USER" == "zli33" ]]; then
    LLAMA_CPP_PATH="/home/nfs/zli33/projects/llama.cpp/"
    MPATH="/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_models/models/llava-v1.6-vicuna-7b/vit/"
    MMPROJ_PATH="/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_models/models/llava-v1.6-vicuna-7b/vit/"
    DPARENT_PATH="/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/datasets/vlm_baseline/annotated/"
else
    LLAMA_CPP_PATH="/home/zonghuan/tudelft/projects/llama.cpp/"
    MPATH="/home/zonghuan/tudelft/projects/large_models/models/llava-v1.6-vicuna-7b/vit/"
    MMPROJ_PATH="/home/zonghuan/tudelft/projects/large_models/models/llava-v1.6-vicuna-7b/vit/"
    DPARENT_PATH="/home/zonghuan/tudelft/projects/datasets/modification/annotated/"
fi

EXEC_NAME="llama-llava-cli"
LLAMA_EXEC=${LLAMA_CPP_PATH}${EXEC_NAME}

MNAME="llava-v1.6-vicuna-7b-7B-F32.gguf"
MODEL=${MPATH}${MNAME}
MODEL_NAME_NO_SUFFIX="${MNAME%.*}"

MMPROJ_NAME="mmproj-model-f16.gguf"
MMPROJ=${MMPROJ_PATH}${MMPROJ_NAME}

DATASET_NAME="ConfLab"
IMG_PATH=${DPARENT_PATH}${DATASET_NAME}/imgs/
IMG_NAME="072_GH020162_00062420_007.jpg"

SIMGLE_IMG=${IMG_PATH}${IMG_NAME}
SIMGLE_IMG_2="/home/zonghuan/tudelft/projects/large_models/samples/toyota.png"

# PROMPT="For the above image, first identify each person in it and assign each of them with an ID. Go ahead with the labelling as participants in the image For each person, provide an estimation of their level of involvement with respect to their corresponding interaction. Rate on a scale between 0 and 1. You don't need to specify an interval. Just give the exact number that you think best matches the involvement level. In the end, generate each person's ID and involvement level in a formatted way like {ID: involvement} and write the formatted output into a file."

# PROMPT="The F-formation is defined as a socio-spatial formation in which people have established and maintain a convex space to which everybody in the gathering has direct, easy and equal access. For the above image, first identify each person in it and assign each of them with an ID. Then, identify how many F-formations could be observed, and output each F-formation with its corresponding members."

#PROMPT="An F-formation is characterized by the mutual locations and head, body orientations of interacting targets, and is defined by the convex O-space they encompass such that each target has unhindered access to its center. A valid F-formation was assumed if the constituent targets were in one of the established patterns, or had direct and unconstrained access to the O-space center in case of large groups. For the given image, first identify each person in it. Then, identify how many F-formations could be observed, and output each F-formation with its corresponding members."

#PROMPT="This image shows the scene of an academic conference from an overhead camera. Each person in the image is assigned a number, which is denoted in red and positioned next to each person. Please annotate conversation groups and write them in brackets. A conversation group means that there is only one conversation taking place among its members, and that all members are focusing on the same topic. A conversation group may include 2 or more people. An example format of annotation is (3, 9, 20), (4, 21), (13, 14). Please ignore the unmarked people. Remember there may be singleton people who are not involved in any conversation group, and you don't need to include them in your annotation."

PROMPT="Describe the two images."

timestamp_filename=$(date +%Y%m%d_%H%M%S)
parentdir="$(dirname "${PWD}")"
output_file=${parentdir}/results/${timestamp_filename}.txt
model_log=${parentdir}/model_logs/${timestamp_filename}.txt

#echo Model name: "$MODEL_NAME_NO_SUFFIX" >> "$output_file"
#echo Prompt: "$PROMPT" >> "$output_file"

# All images in folder
#for img in "$IMG_PATH"*
#do
#    prompting $LLAMA_EXEC $MODEL $MMPROJ "$img" "$PROMPT" "$output_file"
#done

# Single image
#prompting $LLAMA_EXEC $MODEL $MMPROJ $SIMGLE_IMG "$PROMPT" "$output_file" "$model_log"

$LLAMA_EXEC -m $MODEL --mmproj $MMPROJ --image $SIMGLE_IMG --image $SIMGLE_IMG_2 -c 4096 -p "$PROMPT"

