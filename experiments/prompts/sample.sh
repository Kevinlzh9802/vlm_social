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

IMG_PATH="/home/zonghuan/tudelft/projects/large_models/samples/"
IMG_NAME="toyota.png"
IMG=${IMG_PATH}${IMG_NAME}

PROMPT="Can you describe this image? What is it?"

OUTPUT=$($LLAMA_EXEC -m $MODEL --mmproj $MMPROJ --image $IMG -c 4096 -p "$PROMPT")
echo "-----------------------------------------------------------------------------"
echo "${OUTPUT}" >> some.txt

/home/zonghuan/tudelft/projects/llama.cpp/llama-llava-cli -m /home/zonghuan/tudelft/projects/large_models/llava-v1.6-vicuna-7b/vit/llava-v1.6-vicuna-7b-7B-F32.gguf --mmproj /home/zonghuan/tudelft/projects/large_models/llava-v1.6-vicuna-7b/vit/mmproj-model-f16.gguf -p "You are a helpful assistant" -cnv

/home/zonghuan/tudelft/projects/llama.cpp/llama-llava-cli -m /home/zonghuan/tudelft/projects/large_models/models/llava-v1.6-vicuna-7b/vit/llava-v1.6-vicuna-7b-7B-F32.gguf --mmproj /home/zonghuan/tudelft/projects/large_models/models/llava-v1.6-vicuna-7b/vit/mmproj-model-f16.gguf --image "/home/zonghuan/tudelft/projects/large_models/samples/toyota.png" -p "describe the image"

python -m llava.serve.cli \
    --model-path /home/zonghuan/tudelft/projects/large_models/models/llava-v1.5-7b \
    --image-file "https://llava-vl.github.io/static/images/view.jpg" \
    --image-file "https://llava-vl.github.io/static/images/view.jpg"
    --load-4bit
# IN="bla@some.com;john@home.com"
# arrIN=(${OUTPUT})
# arrIN=(${OUTPUT//;/ })
# echo ${arrIN}                  # Output: john@home.com
# write output to file: bash prompt.sh &> output.out