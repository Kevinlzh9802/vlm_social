from utils import load_image
import os
import torch
import json
from gen_prompt import gen_prompt_general, gen_prompt_spec
from utils import print_memory_usage, record_info, get_gallery_images, replace_until_common_base

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
    pixel_values = torch.cat(pixel_values_list, dim=0)
    return pixel_values, num_patches_list, gallery_ids


def iter_by_file(image_files, dataset_path, model, tokenizer, generation_config, output, args):
    prompt_general = gen_prompt_general(args)
    for img_filename in image_files:
        # model, tokenizer, generation_config = load_model(model_path)
        img_path = os.path.join(dataset_path, img_filename)
        if args.visual_strat in ['concat', 'number']:
            # Single image
            pixel_values = load_image(img_path, max_num=12).to(torch.bfloat16).cuda()
            prompt = prompt_general
            response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                           history=None, return_history=True)
        elif args.visual_strat in ['gallery']:
            # Multiple images
            gallery_images = get_gallery_images(img_path)
            if not gallery_images:
                print(f"No gallery images found for {img_filename}")
                continue

            pixel_values, num_patches_list, gallery_ids = process_gallery_images(img_path, gallery_images)

            # Form the prompt
            prompt_spec = gen_prompt_spec(gallery_ids)
            prompt = prompt_general + prompt_spec
            response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                           num_patches_list=num_patches_list,
                                           history=None, return_history=True)
        else:
            raise NotImplementedError

        print_memory_usage()
        record_info(output, img_path, response)

        del pixel_values, response, history
        torch.cuda.empty_cache()


def iter_by_json(metadata_path, model, tokenizer, generation_config, output, args):
    """
    Reads a metadata.json file, loads images, and generates prompting information for a vision-language model.

    Parameters:
    - metadata_path (str): Path to the metadata.json file.
    - load_image (function): Function to load an image and return a tensor.
    """

    # Load the metadata.json file
    with open(metadata_path, 'r') as f:
        metadata_entries = json.load(f)

    prompt_general = gen_prompt_general(args)
    # Loop over every entry in the metadata
    for i, (key, entry) in enumerate(metadata_entries.items()):

        # some metadata.json store absolute paths on local machine,
        # so replace them with relative paths on the cluster
        original_image = replace_until_common_base(entry['original_image'], metadata_path, 'image')
        gallery_image = replace_until_common_base(entry['gallery_image'], metadata_path, 'image')

        # Load the images and process them
        pixel_values1 = load_image(original_image, max_num=12).to(torch.bfloat16).cuda()
        pixel_values2 = load_image(gallery_image, max_num=12).to(torch.bfloat16).cuda()

        # Concatenate pixel values
        pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
        num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

        # Generate the question prompt
        # question = f'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
        # Form the prompt

        # prompt_spec = gen_prompt_spec(gallery_ids)
        prompt_spec = '<image>\n'
        prompt = prompt_general + prompt_spec
        response, history = model.chat(tokenizer, pixel_values, prompt, generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=True)

        print_memory_usage()
        record_info(output, original_image, response)

        del pixel_values, response, history
        torch.cuda.empty_cache()



def main():
    iter_by_json('C:\\Users\\Zonghuan Li\\Downloads\\position_concat\\positional_info.json', 3)

if __name__ == '__main__':
    main()
