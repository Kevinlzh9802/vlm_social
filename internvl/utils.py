import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import psutil
import GPUtil
import time
from datetime import datetime
from PIL import Image
import os

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
    generation_config = dict(max_new_tokens=4096, do_sample=False)
    return model, tokenizer, generation_config

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

def record_info(output, img_path, response):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Write the results to the file
    output.write(f"Image Path: {img_path}\n")
    # output.write(f"Gallery Image Path: {gallery_img}\n")
    output.write(f"Time: {current_time}\n")
    output.write(f"Response: {response}\n")
    output.write("-" * 80 + "\n")
    # print(f"Processed {img_filename} with gallery {os.path.basename(gallery_img)}")