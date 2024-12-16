import numpy as np
import torch
from decord import VideoReader, cpu
import os
import re
import socket
from datetime import datetime
import argparse
import sys
from utils import load_image, print_memory_usage, record_info, extract_x, get_gallery_images, load_model
from gen_prompt import gen_prompt_spec, gen_prompt_general
from group_task import process_gallery_images, iter_by_file, iter_by_json


def get_paths_based_on_hostname():
    # Get the current system's hostname
    hostname = socket.gethostname()

    # Define the regular expression pattern to match hostnames like host1.hpc.tudelft.nl, host2, host3
    daic_pattern = r"^.*\.hpc\.tudelft\.nl$"

    # Check if the hostname matches the pattern
    if re.match(daic_pattern, hostname):
        hostname = 'daic'

    # Define paths for different hostnames
    config = {
        'tud1006233': {
            'dataset_dir': '/home/zonghuan/tudelft/projects/datasets/modification/',
            'model_path': '/home/zonghuan/tudelft/projects/large_models/models/',
            'output_file': '/home/zonghuan/tudelft/projects/vlm_social/internvl/experiments/results'
        },
        'daic': {
            'dataset_dir': '/mnt/zonghuan/datasets/vlm_baseline/',
            'model_path': '/mnt/zonghuan/large_models/models/',
            'output_file': '/mnt/zli33/projects/vlm_social/internvl/experiments/results'
        },
        'default': {
            'dataset_dir': '/mnt/zonghuan/datasets/vlm_baseline/',
            'model_path': '/mnt/zonghuan/large_models/models/',
            'output_file': '/mnt/zli33/projects/vlm_social/internvl/experiments/results'
        }
    }

    # Select the appropriate configuration based on the hostname
    paths = config.get(hostname, config['default'])

    dataset_dir = paths['dataset_dir']
    model_path = paths['model_path']
    output_file = paths['output_file']

    return dataset_dir, model_path, output_file

def use_json(args):
    if args.task == 'locate' and args.visual_strat == 'concat':
        return True
    else:
        return False

# Main function to perform the evaluation
def evaluate(dataset_path, model_path, output_path, args=None):
    # Collect all jpg files in the dataset folder
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    image_files.sort(key=extract_x)

    prompt_general = gen_prompt_general(args)
    model, tokenizer, generation_config = load_model(model_path)
    # Open the output file for writing
    with open(output_path, 'w') as output:
        # Record the model path and the prompt template once
        output.write(f"Model Path: {model_path}\n")
        output.write(f"Prompt Template: {prompt_general}\n\n")

        if use_json(args):
            iter_by_json(image_files, dataset_path, model, tokenizer, generation_config, output, args)
        else:
            iter_by_file(image_files, dataset_path, model, tokenizer, generation_config, output, args)


# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

def check_args(args):
    valid_models = ["InternVL2-1B", "InternVL2-2B", "InternVL2-4B", "InternVL2_5-1B"]
    valid_visual_strats = ["gallery", "concat"]
    valid_tasks = ["fform", "cgroup", "locate"]

    if args.model_name not in valid_models:
        print(f"Error: Invalid model_name '{args.model_name}'. Must be one of {valid_models}.")
        sys.exit(1)

    # Validate visual_strat
    if args.visual_strat not in valid_visual_strats:
        print(f"Error: Invalid visual_strat '{args.visual_strat}'. Must be one of {valid_visual_strats}.")
        sys.exit(1)

    # Validate task
    if args.task not in valid_tasks:
        print(f"Error: Invalid task '{args.task}'. Must be one of {valid_tasks}.")
        sys.exit(1)

def get_rel_dataset_path(args):
    if args.modality == "image":
        if args.visual_strat == "gallery":
            return 'image/gallery_bbox'
        elif args.visual_strat == "concat":
            return 'image/gallery_concat'
        else:
            raise ValueError(f"Invalid visual_strat '{args.visual_strat}'.")
    elif args.modality == "video":
        return "video"

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Evaluation script with model, visual strategy, and task.")

    # Add arguments
    parser.add_argument("--model_name", type=str, required=False, default="InternVL2-2B", help="Name of the model to use.")
    parser.add_argument("--visual_strat", type=str, required=False, default="concat", help="Visual strategy: 'gallery' or 'concat'.")
    parser.add_argument("--task", type=str, required=False, default="fform", help="Task to perform: 'fform' or 'cgroup'.")
    parser.add_argument("--dataset", type=str, required=False, default="conflab", help="Dataset: 'conflab' or 'idiap'.")
    parser.add_argument("--modality", type=str, required=False, default="image", help="Modality: 'image' or 'video'.")
    parser.add_argument("--prompt_strat", type=str, required=False, default="plain", help="Modality: 'image' or 'video'.")

    # Parse arguments
    args = parser.parse_args()
    check_args(args)

    # Create configuration dictionary

    dataset_path, model_path, output_path = get_paths_based_on_hostname()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_name = '_'.join([args.model_name, args.dataset, args.visual_strat, args.task, current_time]) + '.txt'

    dataset_path = os.path.join(dataset_path, args.dataset, get_rel_dataset_path(args))
    model_path = os.path.join(model_path, args.model_name)
    output_path = os.path.join(output_path, output_name)

    evaluate(dataset_path, model_path, output_path, args)

if __name__ == '__main__':
    main()
