from openai import OpenAI
import os
import base64
import requests
import argparse
import platform
import yaml
import datetime
import json
import re
import time

WINDOWS_DEVICE_NAME = 'zonghuan_dell'
SINGLE_IMG_RETRIES = 5


def retrieve_key():
    if platform.system() == 'Windows' and platform.node() == WINDOWS_DEVICE_NAME:
        key_path = 'C:\\Users\\zongh\\Desktop\\personal\\fformation_k2.txt'
    elif platform.system() == 'Linux':
        key_path = '/home/zonghuan/tudelft/projects/large_models/openai_fformation_key1.txt'
    else:
        key_path = None

    with open(key_path, 'r') as file:
        # Read the entire content of the file
        api_key = file.read().strip()
    return api_key


# Function to load configuration from a YAML file
def load_config(file_path):
    # file_path = os.path.abspath(file_path)
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load the configuration

# path = os.getcwd()
# client = OpenAI(
    #     organization='org-rcGHPc6kMx8jf3C9WNtcA5pD',
    #     project='proj_5OUFMg5t5AbkO2xr6ZGngCLW',
    #     api_key=api_key,
    # )
    #
    # stream = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": "Say this is a test"}],
    #     stream=True,
    # )
    # for chunk in stream:
    #     if chunk.choices[0].delta.content is not None:
    #         print(chunk.choices[0].delta.content, end="")

    # Path to your image


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_data_path(cfg):
    if platform.system() == 'Windows':
        parent_path = cfg['dataset']['parent_path_win']
    elif platform.system() == 'Linux':
        parent_path = cfg['dataset']['parent_path_linux']
    else:
        raise ValueError
    dataset_name = cfg['dataset']['name']
    dataset_modality = cfg['dataset']['modality']
    data_path = os.path.join(parent_path, dataset_name, dataset_modality)
    assert isinstance(data_path, str)
    return data_path


def get_exp_name(cfg):
    dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return '_'.join([cfg['dataset']['name'], cfg['dataset']['modality'], cfg['prompt_strategy'], dt])


def get_result_file_name(img_file_name):
    return img_file_name.split('.')[0] + '.json'


def add_response_header(new_key_value, response):
    return {**new_key_value, **response}


def prompt_dataset(cfg):
    api_key = retrieve_key()
    data_path = get_data_path(cfg)
    prompt = cfg['prompt']

    exp_name = get_exp_name(cfg)
    results_path = os.path.join(os.getcwd(), 'results', exp_name)
    os.makedirs(results_path, exist_ok=True)

    # record config
    with open(os.path.join(results_path, 'config.json'), "w") as file:
        json.dump(cfg, file, indent=4)

    for file_name in sorted(os.listdir(data_path)):
        single_path = os.path.join(data_path, file_name)

        headers, payload = prompt_params(single_path, api_key, prompt)
        response = prompt_single(headers, payload)

        response = add_response_header({'instance_id': file_name}, response)
        groups = extract_groups(response)

        # record long response
        # result_file_name = get_result_file_name(file_name)
        with open(os.path.join(results_path, 'full_response.json'), "a") as file:
            json.dump(response, file, indent=4)
            file.write("\n")

        # record short response
        with open(os.path.join(results_path, 'groups.txt'), "a") as text_file:
            text_file.write(str(groups) + '\n')

        print(f'Done with {file_name}')


def prompt_params(single_path, api_key, prompt):
    base64_image = encode_image(single_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 4096
    }
    return headers, payload


def prompt_single(headers, payload):
    response = ''
    for k in range(SINGLE_IMG_RETRIES):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response = response.json()
        except Exception as e:
            continue
        if exceed_rate_limit(response):
            time.sleep(k*k + 1)
            print('Exceeding rate limit. Sleeping for retry.')
        else:
            break
    return response


def exceed_rate_limit(response):
    return ('error' in response.keys()) and (response['error']['code'] == 'rate_limit_exceeded')


def string_to_tuple(s):
    try:
        # Convert to tuple of integers
        return tuple(int(item.strip()) for item in s.strip("()").split(","))
    except ValueError:
        # Return None if conversion fails
        return None


def find_and_convert_tuples(text):
    # Find all contents wrapped in parentheses
    matches = re.findall(r'\([^()]*\)', text)
    # Convert each match to a tuple, ignoring non-numeric tuples
    tuples = []
    for match in matches:
        result = string_to_tuple(match)
        if result is not None:
            tuples.append(result)
    return tuples


def extract_groups(response):
    try:
        short_response = response['choices'][0]['message']['content']
    except:
        short_response = ''
        print('Warning: could not deal with response')

    return find_and_convert_tuples(short_response)


def main():
    parser = argparse.ArgumentParser(description="A program with optional arguments.")
    parser.add_argument("-config", type=str, default="default1", help="The first argument (optional)")

    args = parser.parse_args()
    config = load_config(args.config)
    prompt_dataset(config)


if __name__ == '__main__':
    main()
