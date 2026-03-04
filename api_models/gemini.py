import argparse
from pathlib import Path
from typing import Optional

try:
    from google import genai
except ImportError:
    genai = None

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_PROMPT = "Describe what is happening in this video."
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "api_keys" / "gemini.txt"


def load_api_key(api_key_path: Path = DEFAULT_API_KEY_PATH) -> str:
    with open(api_key_path, "r", encoding="utf-8") as file:
        api_key = file.read().strip()

    if not api_key:
        raise ValueError(f"Gemini API key file is empty: {api_key_path}")
    return api_key


def get_client(api_key: Optional[str] = None, api_key_path: Path = DEFAULT_API_KEY_PATH):
    key = api_key or load_api_key(api_key_path=api_key_path)
    return genai.Client(api_key=key)


def generate_video_response(
    video_path: str,
    prompt: str = DEFAULT_PROMPT,
    model_name: str = DEFAULT_MODEL,
    client=None,
) -> str:
    """Upload a video and return Gemini's text response."""
    if client is None:
        client = get_client()

    uploaded_video = client.files.upload(file=video_path)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, uploaded_video],
    )

    text = getattr(response, "text", None)
    if not text:
        raise ValueError(f"Gemini returned an empty response for video: {video_path}")
    return text


def main():
    parser = argparse.ArgumentParser(description="Run Gemini inference on a single video.")
    parser.add_argument("--video-path", type=str, default="data/clip_11.mp4", help="Path to a video file.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt for Gemini.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name.")
    args = parser.parse_args()

    response_text = generate_video_response(
        video_path=args.video_path,
        prompt=args.prompt,
        model_name=args.model,
    )
    print(response_text)


if __name__ == "__main__":
    main()
