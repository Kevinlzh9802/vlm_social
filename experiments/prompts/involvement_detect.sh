#!/bin/sh

LLAMA_CPP_PATH="/home/zonghuan/tudelft/projects/llama.cpp/"
EXEC_NAME="llama-llava-cli"
LLAMA_EXEC=${LLAMA_CPP_PATH}${EXEC_NAME}

MPATH="/home/zonghuan/tudelft/projects/large_models/llava-v1.6-vicuna-7b/vit/"
MNAME="llava-v1.6-vicuna-7b-7B-F32.gguf"
MODEL=${MPATH}${MNAME}

MMPROJ_PATH="/home/zonghuan/tudelft/projects/large_models/llava-v1.6-vicuna-7b/vit/"
MMPROJ_NAME="mmproj-model-f16.gguf"
MMPROJ=${MMPROJ_PATH}${MMPROJ_NAME}

IMG_PATH="/home/zonghuan/tudelft/projects/datasets/conflab/modification/"
IMG_NAME="f000005.png"
IMG=${IMG_PATH}${IMG_NAME}

PROMPT="For the above image, first identify each person in it and assign each of them with an ID. Go ahead with the labelling as participants in the image For each person, provide an estimation of their level of involvement with respect to their corresponding interaction. Rate on a scale between 0 and 1. You don't need to specify an interval. Just give the exact number that you think best matches the involvement level. In the end, generate each person's ID and involvement level in a formatted way like {ID: involvement} and write the formatted output into a file."

OUTPUT=$($LLAMA_EXEC -m $MODEL --mmproj $MMPROJ --image $IMG -c 4096 -p "$PROMPT")
echo "-----------------------------------------------------------------------------"
echo "${OUTPUT}"

# echo ${arrIN}                  # Output: john@home.com
# write output to file: bash prompt.sh &> output.out
