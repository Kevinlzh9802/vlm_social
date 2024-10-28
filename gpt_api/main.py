from openai import OpenAI
import os
import base64
import requests

key_path = '/home/zonghuan/tudelft/projects/large_models/openai_fformation_key1.txt'
with open(key_path, 'r') as file:
    # Read the entire content of the file
    api_key = file.read().strip()

# path = os.getcwd()

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def main():
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
    image_path = "/home/zonghuan/tudelft/projects/large_models/samples/toyota.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

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
                        "text": "What’s in this image?"
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

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())


if __name__ == '__main__':
    main()