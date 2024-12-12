import numpy as np
# import torch
# import torchvision.transforms as T
# from decord import VideoReader, cpu
from PIL import Image
# from torchvision.transforms.functional import InterpolationMode
# from transformers import AutoModel, AutoTokenizer
import os
import re
import socket
from datetime import datetime
import psutil
# import GPUtil
import time
import argparse
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    mean, std = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

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

def extract_x(filename):
    # Split the filename at the first underscore and extract the first part
    x_part = filename.split('_')[0]
    # Try to convert the first part to an integer, default to infinity if conversion fails
    try:
        return int(x_part)
    except ValueError:
        return float('inf')

def get_gallery_images(image_path):
    base_name = os.path.basename(image_path).split('.')[0]
    dataset_dir = os.path.dirname(image_path)
    gallery_dir = os.path.join(dataset_dir, "gallery", base_name)
    if os.path.exists(gallery_dir):
        return [os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.endswith('.jpg')]
    return []

def get_cpu_memory_usage():
    """Get the current CPU memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Convert to MB


def get_gpu_memory_usage():
    """Get the current GPU memory usage in MB."""
    gpus = GPUtil.getGPUs()
    gpu_memory = []
    for gpu in gpus:
        gpu_memory.append(f"GPU {gpu.id}: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
    return gpu_memory


def print_memory_usage(interval=5, duration=60):
    """Print CPU and GPU memory usage at regular intervals."""
    # start_time = time.time()
    # while (time.time() - start_time) < duration:
    cpu_memory = get_cpu_memory_usage()
    gpu_memory = get_gpu_memory_usage()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[{current_time}]")
    print(f"CPU Memory Usage: {cpu_memory:.2f} MB")
    for gpu_info in gpu_memory:
        print(f"{gpu_info}")

        # time.sleep(interval)

def load_model(model_path):
    """
    Load the model and tokenizer from the specified model path.
    """
    print('Loading model...')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    print('Model loaded successfully.')
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    return model, tokenizer, generation_config

def gen_prompt_general():
    converse_group_prompt = "You will be given an original image and several gallery images. The original images ia about a scene captured from an overhead camera, and each gallery image captures a person in the scene. Each person is associated with an ID. Your job is to determine which people are within the same conversation group in the original scene (i.e. by considering who is talking with whom). Only consider people given in the gallery images. A conversation group may include 2 or more people. Give the response by grouping the IDs of people in parentheses. An example format of answer is (3, 9, 20), (4, 21), (13, 14). Remember there may be singleton people who are not involved in any conversation group, and you don't need to include them in your answer."

    converse_group_prompt_single_number = "Suppose you're an expert in annotating conversation groups. This image shows the scene of an academic conference from an overhead camera. Each person in the image is assigned a number, which is denoted in red and positioned next to each person. Please annotate conversation groups and write them in brackets. A conversation group means that there is only one conversation taking place among its members, and that all members are focusing on the same topic. A conversation group may include 2 or more people. Please ignore the unmarked people. Remember there may be singleton people who are not involved in any conversation group, and you don't need to include them in your annotation. Let's think step by step by considering people's behavioral cues. For example, two people facing each other might indicate they are in the same conversation group, while two people back-to-back or very distant from each other are rather unlikely to be in the same group. An example format of annotation is [(3, 9, 20), (4, 21), (13, 14)]. Please output your response in this format."

    converse_group_prompt_gallery_concat = "Suppose you're an expert in annotating conversation groups. This image on the left shows the scene of an academic conference from an overhead camera. The small images on the right side are the gallery, which consists of some people cropped from the left side. Each person is assigned a number, which is denoted in red and positioned under their corresponding gallery image. Please annotate conversation groups and write them in parentheses. Use IDs of the people to represent them. A conversation group means that there is only one conversation taking place among its members, and that all members are focusing on the same topic. A conversation group may include 2 or more people. Please ignore people not included in the gallery. Remember there may be singleton people who are not involved in any conversation group, and you don't need to include them in your annotation. "

    step_prompt = "Let's think step by step by considering people's behavioral cues. "
    cue_hint = "For example, two people facing each other might indicate they are in the same conversation group, while two people back-to-back or very distant from each other are rather unlikely to be in the same group. "
    example = "An example format of annotation is [(3, 9, 20), (4, 21), (13, 14)]. Please output your response in a format similar to this. "

    converse_group_prompt_gallery_concat_simple = "Who and whom are talking together? Look at the image on the left side and refer to the right side for IDs of people. Format the output in parentheses. For example, [(x1, x2, x3), (x4, x5)], where x1, ..., x5 are people's IDs. This example annotation means that you believe x1, x2, x3 are talking together, while x4 and x5 are talking together. Now, provide your answer regarding this image."

    fformation_prompt = "The F-formation is defined as a socio-spatial formation, where every member of it has direct, easy and equal access. Look at the image on the left side, what F-formations do you observe in it? Refer to the right side for IDs of people. Output several brackets, each containing the IDs of people within one F-formation. "
    example_2 = "For example, your output should look like [(x1, x2, x3), (x4, x5)]. This sample answer means that you believe x1, x2, x3 are in one F-formation, while x4 and x5 are in another, where x1, ..., x5 are people's IDs. Now, provide your answer regarding this image. Do not simply repeat the sample answer! Replace the IDs with the read IDs of people in the image!"

    return converse_group_prompt_gallery_concat_simple


def gen_prompt_spec(img_path, gallery_ids):
    """
    Forms a prompt for the model that includes the original image and multiple gallery images,
    each labeled with its corresponding ID.

    Args:
        img_path (str): Path to the original image.
        gallery_ids (list): List of gallery image IDs (filenames).

    Returns:
        str: The formatted prompt string.
    """
    prompt_lines = [f"Original Image: <image>"]

    # Add the original image with its label
    # Add each gallery image with its ID and placeholder
    for _, gallery_filename in enumerate(gallery_ids, start=1):
        person_id = gallery_filename.split('.')[0]
        prompt_lines.append(f"Person of ID {person_id}: <image>")

    # # Add the instruction to describe all the images in detail
    # prompt_lines.append("\nDescribe the images in detail.")

    # Join all lines into a single prompt string
    prompt = "\n".join(prompt_lines)
    return prompt


def process_gallery_images(img_path, gallery_images):
    """
    Loop over gallery images and evaluate the model.
    """
    if not gallery_images:
        print(f"No gallery images found for {img_path}")
        return

    # Load the original image
    pixel_values_list = [load_image(img_path, max_num=12).to(torch.bfloat16).cuda()]
    num_patches_list = [pixel_values_list[0].size(0)]
    gallery_ids = []
    # num_patches_list = []
    for gallery_img in gallery_images:
        try:
            # Initialize with the number of patches in the original image
            # Load all gallery images and append to the list
            pixel_values_gallery = load_image(gallery_img, max_num=2).to(torch.bfloat16).cuda()
            pixel_values_list.append(pixel_values_gallery)
            num_patches_list.append(pixel_values_gallery.size(0))
            gallery_id = os.path.basename(gallery_img)
            gallery_ids.append(gallery_id)
        except Exception as e:
            print(f"Error processing {os.path.basename(img_path)} with gallery {os.path.basename(gallery_img)}: {e}")

            # Concatenate all the pixel values
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list, gallery_ids


# Main function to perform the evaluation
def evaluate(dataset_path, model_path, output_path, config=None):
    # Collect all jpg files in the dataset folder
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    image_files.sort(key=extract_x)

    # model, tokenizer, generation_config = load_model(model_path)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_name = current_time + '.txt'
    output_file = os.path.join(output_path, output_name)
    # Open the output file for writing
    with open(output_file, 'w') as output:
        # Record the model path and the prompt template once

        prompt_general = gen_prompt_general()
        output.write(f"Model Path: {model_path}\n")
        output.write(f"Prompt Template: {prompt_general}\n\n")

        for img_filename in image_files:
            model, tokenizer, generation_config = load_model(model_path)
            img_path = os.path.join(dataset_path, img_filename)
            prompt_spec = ''
            pixel_values_main = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
            num_patches_list = [pixel_values_main[0].size(0)]
            # gallery_images = get_gallery_images(img_path)
            # if not gallery_images:
            #     print(f"No gallery images found for {img_filename}")
            #     continue
            #
            # pixel_values, num_patches_list, gallery_ids = process_gallery_images(img_path, gallery_images)
            #
            # # Form the prompt
            # prompt_spec = gen_prompt_spec(img_path, gallery_ids)

            prompt = prompt_general + prompt_spec
            print_memory_usage()
            pixel_values = pixel_values_main
            # Get response from the model
            response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                           # num_patches_list=num_patches_list,
                                           history=None, return_history=True)

            # Get the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Write the results to the file
            output.write(f"Image Path: {img_path}\n")
            # output.write(f"Gallery Image Path: {gallery_img}\n")
            output.write(f"Time: {current_time}\n")
            output.write(f"Response: {response}\n")
            output.write("-" * 80 + "\n")
            # print(f"Processed {img_filename} with gallery {os.path.basename(gallery_img)}")

            del pixel_values, response, history
            torch.cuda.empty_cache()


# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.

def check_args(args):
    valid_models = ["internvl1b", "internvl2b", "internvl4b"]
    valid_visual_strats = ["multi", "concat"]
    valid_tasks = ["fform", "cgroup"]

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
    if args.visual_strat == "multi":
        return 'imgs_gallery'
    elif args.visual_strat == "concat":
        return 'imgs_gallery_concat'
    else:
        raise ValueError(f"Invalid visual_strat '{args.visual_strat}'.")

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Evaluation script with model, visual strategy, and task.")

    # Add arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--visual_strat", type=str, required=True, help="Visual strategy: 'multi' or 'concat'.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform: 'fform' or 'cgroup'.")
    parser.add_argument("--dataset", type=str, required=False, default="conflab", help="Task to perform: 'fform' or 'cgroup'.")

    # Parse arguments
    args = parser.parse_args()
    check_args(args)

    # Create configuration dictionary
    cfg = {
        "model_name": args.model_name,
        "visual_strat": args.visual_strat,
        "task": args.task
    }

    dataset_path, model_path, output_path = get_paths_based_on_hostname()
    output_name = '_'.join([args.modle_name, args.dataset, args.visual_strat, args.task]) + '.txt'

    dataset_path = os.path.join(dataset_path, args.dataset, get_rel_dataset_path(args))
    model_path = os.path.join(model_path, args.model_name)
    output_path = os.path.join(output_path, output_name)

    evaluate(dataset_path, model_path, output_path, cfg)

if __name__ == '__main__':
    main()
