import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
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

# def multi_img_query(model, images: list, tokenizer, generation_config):
#     pixel_values1 = load_image(img_toyota, max_num=12).to(torch.bfloat16).cuda()
#     pixel_values2 = load_image(img_nvidia, max_num=12).to(torch.bfloat16).cuda()
#     pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
#     num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
#
#     question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
#     response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                    num_patches_list=num_patches_list,
#                                    history=None, return_history=True)
#     print(f'User: {question}\nAssistant: {response}')


# If you have an 80G A100 GPU, you can put the entire model on a single GPU.
# Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
def main():
    model_path = '/home/zonghuan/tudelft/projects/large_models/models/InternVL2-4B'
    model_path = '/mnt/zonghuan/large_models/models/InternVL2-4B'
    # model_path = '/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/large_models/models/InternVL2-4B'
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)

    img_bird = '/home/zonghuan/tudelft/projects/large_models/samples/bird.png'
    img_toyota = '/home/zonghuan/tudelft/projects/large_models/samples/toyota.png'
    img_nvidia = '/home/zonghuan/tudelft/projects/large_models/samples/nvidia.png'

    img_toyota = '/mnt/zonghuan/datasets/sample/toyota.png'
    img_nvidia = '/mnt/zonghuan/datasets/sample/toyota.png'
    #/tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/datasets/vlm_baseline/conflab_bbox_sample/gallery
    # /tudelft.net/staff-bulk/ewi/insy/SPCLab/zonghuan/datasets/sample
    # for subfolder in os.listdir(folder_path):
    #     subfolder_path = os.path.join(folder_path, subfolder)

    img_main = '/home/zonghuan/tudelft/projects/datasets/modification/conflab_bbox/000000_cam8_seg6.jpg'
    img_gallery_1 = '/home/zonghuan/tudelft/projects/datasets/modification/conflab_bbox/gallery/000000_cam8_seg6/12.jpg'

    # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
    pixel_values1 = load_image(img_toyota, max_num=12).to(torch.bfloat16).cuda()
    pixel_values2 = load_image(img_nvidia, max_num=12).to(torch.bfloat16).cuda()
    pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
    num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

    # question = 'Question: The second images shows a person. Where is this person in the first image? \nSelect from the following choices: (A)Top-left\n (B)Top-right\n (C)Bottom-left\n (D)Bottom-right. Image-1: <image>\nImage-2: <image>\n'
    # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
    #                                num_patches_list=num_patches_list,
    #                                history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

    question = 'What are the differences between these two companies?Image1: <image>\n, Image 2: <image>\n'
    response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                   num_patches_list=num_patches_list,
                                   history=None, return_history=True)
    print(f'User: {question}\nAssistant: {response}')

if __name__ == '__main__':
    main()
