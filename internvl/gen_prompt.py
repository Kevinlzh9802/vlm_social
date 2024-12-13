import yaml

def gen_prompt_general(args):
    # with open('/mnt/zli33/projects/vlm_social/internvl/prompts.yaml', 'r') as file:
    with open('prompts.yaml', 'r') as file:
        prompts = yaml.safe_load(file)
    task_prompt = prompts[args.task]
    basic = task_prompt['def']

    if args.visual_strat in ['concat', 'number']:
        visual_strat_prompt = task_prompt['single']
    elif args.visual_strat in ['gallery']:
        visual_strat_prompt = task_prompt['multi']
    else:
        raise ValueError('Visual strategy not recognized')

    scene_desc = (visual_strat_prompt[args.visual_strat]['modal_desc'] +
                  visual_strat_prompt[args.visual_strat]['scene_desc'] +
                  visual_strat_prompt[args.visual_strat]['task_desc'])
    prompt_plain = scene_desc + basic + task_prompt['output_instruct'] + prompts['img_token']
    prompt_candidates = {
        'plain': prompt_plain,
        'role': task_prompt['role'] + prompt_plain,
        'step': prompts['step_prompt'] + prompt_plain,
    }
    return prompt_candidates[args.prompt_strat]


def gen_prompt_spec(gallery_ids, args=None):
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
