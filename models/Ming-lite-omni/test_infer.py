import os
import time
import torch
from transformers import AutoProcessor
from transformers.utils import is_flash_attn_2_available

from modeling_bailingmm import BailingMMNativeForConditionalGeneration

def generate(messages, processor, model):
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_inputs, video_inputs, audio_inputs = processor.process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        audios=audio_inputs,
        return_tensors="pt",
    ).to(model.device)

    for k in inputs.keys():
        if k == "pixel_values" or k == "pixel_values_videos" or k == "audio_feats":
            inputs[k] = inputs[k].to(dtype=torch.bfloat16)
    
    srt_time = time.time()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=False,
        eos_token_id=processor.gen_terminator,
    )
    generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    print(f"Generate time: {(time.time() - srt_time):.2f}s")


if __name__ == '__main__':
    processor = AutoProcessor.from_pretrained(".", trust_remote_code=True)
    model_path = "."
    attn_impl = "flash_attention_2" if is_flash_attn_2_available() else "eager"
    model = BailingMMNativeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )

    vision_path = "/input/zhangqinglong.zql/assets/"

    """
    Disabled examples below (kept for reference):
    # qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "请详细介绍鹦鹉的生活习性。"}
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

    # # image qa
    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "image", "image": os.path.join(vision_path, "flowers.jpg")},
    #             {"type": "text", "text": "What kind of flower is this?"},
    #         ],
    #     },
    # ]
    # generate(messages=messages, processor=processor, model=model)

    # video qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": os.path.join(vision_path, "yoga.mp4"),"max_frames":128, "sample": "uniform"},
                {"type": "text", "text": "Explain what you see?"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)
    """
    # multi-turn chat
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "中国的首都是哪里？"},
            ],
        },
        {
            "role": "ASSISTANT",
            "content": [
                {"type": "text", "text": "北京"},
            ],
        },
        {
            "role": "HUMAN",
            "content": [
                {"type": "text", "text": "它的占地面积是多少？有多少常住人口？"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)

    # messages = [
    #     {
    #         "role": "HUMAN",
    #         "content": [
    #             {"type": "text", "text": "Please recognize the language of this speech and transcribe it. Format: oral."},
    #             {"type": "audio", "audio": '/input/dongli.xq/BAC009S0915W0292.wav'},
    #         ],
    #     },
    # ]
    # generate(messages=messages, processor=processor, model=model)
    # """

    # video + audio qa
    messages = [
        {
            "role": "HUMAN",
            "content": [
                {"type": "video", "video": "/scratch/zli33/data/gestalt_bench/sample/clip_11.mp4", "max_frames": 128, "sample": "uniform"},
                {"type": "audio", "audio": "/scratch/zli33/data/gestalt_bench/sample/clip_11.wav"},
                {"type": "text", "text": "describe what you see in this video"},
            ],
        },
    ]
    generate(messages=messages, processor=processor, model=model)
