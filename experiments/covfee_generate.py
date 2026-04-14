#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def build_pairs(root_dir: Path) -> list[dict[str, str]]:
	pairs: list[dict[str, str]] = []

	for folder in sorted(p for p in root_dir.iterdir() if p.is_dir()):
		mp4_by_stem = {p.stem: p for p in folder.glob("*.mp4")}
		wav_by_stem = {p.stem: p for p in folder.glob("*.wav")}

		for stem in sorted(set(mp4_by_stem) & set(wav_by_stem)):
			video_path = mp4_by_stem[stem].relative_to(root_dir).as_posix()
			audio_path = wav_by_stem[stem].relative_to(root_dir).as_posix()
			pairs.append({"id": len(pairs), "video": f"http://localhost:5000/api/media/gestalt_bench/{video_path}", "audio": f"http://localhost:5000/api/media/gestalt_bench/{audio_path}"})

	return pairs


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Generate a JSON file containing {video, audio} entries for .mp4/.wav "
			"files with the same name in each subfolder."
		)
	)
	parser.add_argument(
		"--root",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Root directory containing subfolders of media files.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).resolve().parent / "gestalt_pairs.json",
		help="Path for output JSON file.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	root_dir = args.root.resolve()
	output_path = args.output.resolve()

	pairs = build_pairs(root_dir)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as f:
		json.dump(pairs, f, indent=2)
		f.write("\n")

	print(f"Wrote {len(pairs)} pairs to {output_path}")


if __name__ == "__main__":
	main()