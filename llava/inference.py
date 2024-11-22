from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
import torch
import requests
from PIL import Image


def main():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )


    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id, pad_token="<pad>")
    model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")


    image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
    image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)


    prompts = [
                "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
                "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
                # "USER: <image>\nPlease describe this image\nASSISTANT:",
    ]

    inputs = processor(prompts, images=image1, padding=True, return_tensors="pt").to("cuda")
    for k,v in inputs.items():
      print(k,v.shape)

    output = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text in generated_text:
        print(text.split("ASSISTANT:")[-1])

if __name__ == "__main__":
    main()
