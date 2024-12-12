import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
import re
import socket
from datetime import datetime
import psutil
import GPUtil
import time

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

def form_prompt():
    return "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail."

# Main function to perform the evaluation
def evaluate(dataset_path, model_path, output_path):
    # Collect all jpg files in the dataset folder
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    image_files.sort(key=extract_x)

    #
    print('before load')
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    output_file = os.path.join(output_path, 'output.txt')
    # Open the output file for writing

    with open(output_file, 'w') as output:
        # Record the model path and the prompt template once

        prompt_template = form_prompt()
        output.write(f"Model Path: {model_path}\n")
        output.write(f"Prompt Template: {prompt_template}\n\n")

        for img_filename in image_files:
            img_path = os.path.join(dataset_path, img_filename)
            gallery_images = get_gallery_images(img_path)

            if not gallery_images:
                print(f"No gallery images found for {img_filename}")
                continue

            print('before iterate gallery')
            # Load the original image and one gallery image at a time for evaluation
            for gallery_img in gallery_images:
                # print_memory_usage()
                # print(gallery_img)
                try:
                    # Load images
                    pixel_values1 = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
                    pixel_values2 = load_image(gallery_img, max_num=12).to(torch.bfloat16).cuda()
                    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
                    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

                    # Form the prompt
                    question = form_prompt()

                    # Get the current time
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    print('after converting pixel values')
                    print_memory_usage()
                    # Get response from the model
                    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                                   num_patches_list=num_patches_list,
                                                   history=None, return_history=True)

                    # Write the results to the file
                    output.write(f"Image Path: {img_path}\n")
                    output.write(f"Gallery Image Path: {gallery_img}\n")
                    output.write(f"Time: {current_time}\n")
                    output.write(f"Response: {response}\n")
                    output.write("-" * 80 + "\n")

                    print(f"Processed {img_filename} with gallery {os.path.basename(gallery_img)}")

                    del pixel_values1, pixel_values2, pixel_values, response, history
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error processing {img_filename} with gallery {os.path.basename(gallery_img)}: {e}")

# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
def main():
    dataset_dir, model_path, output_path = get_paths_based_on_hostname()
    dataset_name = 'conflab_bbox_sample'
    model_name = 'InternVL2-2B'

    dataset_dir = os.path.join(dataset_dir, dataset_name)
    model_path = os.path.join(model_path, model_name)

    print('before evaluation')
    evaluate(dataset_dir, model_path, output_path)

if __name__ == '__main__':
    main()
