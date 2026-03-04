import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from api_models.gemini import DEFAULT_MODEL, DEFAULT_PROMPT, generate_video_response, get_client
except ModuleNotFoundError:
    import sys

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from api_models.gemini import DEFAULT_MODEL, DEFAULT_PROMPT, generate_video_response, get_client

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def discover_group_folders(dataset_path: Path) -> List[Path]:
    return sorted([p for p in dataset_path.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def discover_videos(group_path: Path) -> List[Path]:
    videos = []
    for path in group_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)
    videos.sort(key=lambda p: str(p).lower())
    return videos


def initialize_results(
    dataset_path: Path,
    model_name: str,
    prompt: str,
    max_retries: int,
) -> Dict:
    groups = []
    for group_path in discover_group_folders(dataset_path):
        videos = discover_videos(group_path)
        video_entries = []
        for video_path in videos:
            video_entries.append(
                {
                    "video_name": video_path.name,
                    "video_path": str(video_path.resolve()),
                    "relative_path_in_group": str(video_path.relative_to(group_path)),
                    "status": "pending",
                    "attempts": 0,
                    "response_text": None,
                    "last_error": None,
                    "attempt_history": [],
                }
            )

        groups.append(
            {
                "group_name": group_path.name,
                "group_path": str(group_path.resolve()),
                "status": "pending" if video_entries else "empty",
                "videos": video_entries,
                "stats": {
                    "total": len(video_entries),
                    "succeeded": 0,
                    "failed": 0,
                    "retry_pending": 0,
                    "pending": len(video_entries),
                },
            }
        )

    return {
        "dataset_path": str(dataset_path.resolve()),
        "model_name": model_name,
        "prompt": prompt,
        "max_retries": max_retries,
        "started_at_utc": utc_now_iso(),
        "completed_at_utc": None,
        "rounds_executed": 0,
        "summary": {
            "groups_total": len(groups),
            "groups_succeeded": 0,
            "groups_failed": 0,
            "groups_partial_failed": 0,
            "groups_empty": 0,
            "videos_total": sum(len(group["videos"]) for group in groups),
            "videos_succeeded": 0,
            "videos_failed": 0,
            "videos_retry_pending": 0,
            "videos_pending": sum(len(group["videos"]) for group in groups),
        },
        "groups": groups,
    }


def refresh_group_stats(group: Dict) -> None:
    total = len(group["videos"])
    succeeded = sum(1 for v in group["videos"] if v["status"] == "success")
    failed = sum(1 for v in group["videos"] if v["status"] == "failed")
    retry_pending = sum(1 for v in group["videos"] if v["status"] == "retry_pending")
    pending = sum(1 for v in group["videos"] if v["status"] == "pending")

    group["stats"] = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "retry_pending": retry_pending,
        "pending": pending,
    }

    if total == 0:
        group["status"] = "empty"
    elif succeeded == total:
        group["status"] = "success"
    elif failed == total:
        group["status"] = "failed"
    elif failed > 0:
        group["status"] = "partial_failed"
    elif retry_pending > 0:
        group["status"] = "retry_pending"
    else:
        group["status"] = "pending"


def refresh_summary(results: Dict) -> None:
    groups = results["groups"]
    videos_total = 0
    videos_succeeded = 0
    videos_failed = 0
    videos_retry_pending = 0
    videos_pending = 0

    groups_succeeded = 0
    groups_failed = 0
    groups_partial_failed = 0
    groups_empty = 0

    for group in groups:
        refresh_group_stats(group)
        videos_total += group["stats"]["total"]
        videos_succeeded += group["stats"]["succeeded"]
        videos_failed += group["stats"]["failed"]
        videos_retry_pending += group["stats"]["retry_pending"]
        videos_pending += group["stats"]["pending"]

        if group["status"] == "success":
            groups_succeeded += 1
        elif group["status"] == "failed":
            groups_failed += 1
        elif group["status"] == "partial_failed":
            groups_partial_failed += 1
        elif group["status"] == "empty":
            groups_empty += 1

    results["summary"] = {
        "groups_total": len(groups),
        "groups_succeeded": groups_succeeded,
        "groups_failed": groups_failed,
        "groups_partial_failed": groups_partial_failed,
        "groups_empty": groups_empty,
        "videos_total": videos_total,
        "videos_succeeded": videos_succeeded,
        "videos_failed": videos_failed,
        "videos_retry_pending": videos_retry_pending,
        "videos_pending": videos_pending,
    }


def persist_results(output_json: Path, results: Dict) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def collect_retry_targets(results: Dict, max_retries: int) -> List[Tuple[int, int]]:
    targets = []
    for group_index, group in enumerate(results["groups"]):
        for video_index, video in enumerate(group["videos"]):
            if video["status"] == "success":
                continue
            if video["attempts"] >= max_retries:
                continue
            targets.append((group_index, video_index))
    return targets


def run_batch_inference(
    dataset_path: Path,
    output_json: Path,
    model_name: str,
    prompt: str,
    max_retries: int,
    retry_delay_seconds: float,
) -> Dict:
    results = initialize_results(
        dataset_path=dataset_path,
        model_name=model_name,
        prompt=prompt,
        max_retries=max_retries,
    )
    persist_results(output_json=output_json, results=results)

    client = None
    round_count = 0

    while True:
        targets = collect_retry_targets(results=results, max_retries=max_retries)
        if not targets:
            break

        if client is None:
            client = get_client()

        round_count += 1
        for group_index, video_index in targets:
            video = results["groups"][group_index]["videos"][video_index]
            video_path = video["video_path"]
            video["attempts"] += 1
            attempt = video["attempts"]

            try:
                response_text = generate_video_response(
                    video_path=video_path,
                    prompt=prompt,
                    model_name=model_name,
                    client=client,
                )
                video["status"] = "success"
                video["response_text"] = response_text
                video["last_error"] = None
                video["attempt_history"].append(
                    {
                        "attempt": attempt,
                        "status": "success",
                        "timestamp_utc": utc_now_iso(),
                    }
                )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                video["last_error"] = error_message
                if attempt < max_retries:
                    video["status"] = "retry_pending"
                else:
                    video["status"] = "failed"
                video["attempt_history"].append(
                    {
                        "attempt": attempt,
                        "status": video["status"],
                        "timestamp_utc": utc_now_iso(),
                        "error": error_message,
                        "traceback": traceback.format_exc(),
                    }
                )

            refresh_summary(results)
            persist_results(output_json=output_json, results=results)
            if video["status"] != "success" and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)

    results["rounds_executed"] = round_count
    results["completed_at_utc"] = utc_now_iso()
    refresh_summary(results)
    persist_results(output_json=output_json, results=results)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Gemini inference over grouped video folders.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path containing video group folders.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="results/gemini_batch_results.json",
        help="Output JSON path for statuses and responses.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt to send with each video.")
    parser.add_argument("--max-retries", type=int, default=3, help="Max attempts per video.")
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=2.0,
        help="Delay after each failed attempt before continuing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_json = Path(args.output_json)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be greater than 0.")

    run_batch_inference(
        dataset_path=dataset_path,
        output_json=output_json,
        model_name=args.model,
        prompt=args.prompt,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print(f"Finished. Results saved to: {output_json.resolve()}")


if __name__ == "__main__":
    main()
