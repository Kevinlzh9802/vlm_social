import argparse
import json
import random
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench")
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT / "human_eval" / "samples"
SINGLE_GROUP_PATTERN = re.compile(r"^d(?P<dialogue>\d+)u(?P<utterance>\d+)$")

DATASET_CONFIGS = {
    "mintrec2": {
        "sample_count": 160,
        "batch_count": 4,
        "group_sizes": (1, 2, 3),
    },
    "meld": {
        "sample_count": 160,
        "batch_count": 4,
        "group_sizes": (1, 2, 3),
    },
    "seamless_interaction": {
        "sample_count": 80,
        "batch_count": 2,
        "group_sizes": (1, 2),
    },
}

GROUP_FOLDER_NAMES = {
    1: "1-utt_group",
    2: "2-utt_group",
    3: "3-utt_group",
}


@dataclass(frozen=True)
class SampleCandidate:
    dataset: str
    dialogue_id: int
    end_utterance_id: int
    anchor_name: str
    group_dirs: dict[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly select grouped human-evaluation samples, copy them into "
            "batched folders, and create zip archives for each batch."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Dataset parent folder containing mintrec2, meld, and seamless_interaction.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Output folder where copied batches, zip files, and summaries are written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing output root before writing new selections.",
    )
    return parser.parse_args()


def parse_anchor_name(name: str) -> tuple[int, int] | None:
    match = SINGLE_GROUP_PATTERN.fullmatch(name)
    if match is None:
        return None
    return int(match.group("dialogue")), int(match.group("utterance"))


def build_group_name(dialogue_id: int, end_utterance_id: int, group_size: int) -> str:
    start_utterance_id = end_utterance_id - group_size + 1
    if group_size == 1:
        return f"d{dialogue_id}u{end_utterance_id}"
    return f"d{dialogue_id}u{start_utterance_id}-u{end_utterance_id}"


def discover_candidates(dataset_root: Path, dataset_name: str, group_sizes: tuple[int, ...]) -> list[SampleCandidate]:
    context_root = dataset_root / "context"
    single_group_root = context_root / GROUP_FOLDER_NAMES[1]
    if not single_group_root.is_dir():
        raise FileNotFoundError(f"Missing group folder: {single_group_root}")

    candidates: list[SampleCandidate] = []
    for sample_dir in sorted(path for path in single_group_root.iterdir() if path.is_dir()):
        parsed = parse_anchor_name(sample_dir.name)
        if parsed is None:
            continue

        dialogue_id, end_utterance_id = parsed
        group_dirs: dict[int, str] = {}
        missing_group = False

        for group_size in group_sizes:
            start_utterance_id = end_utterance_id - group_size + 1
            if start_utterance_id < 0:
                missing_group = True
                break

            group_name = build_group_name(dialogue_id, end_utterance_id, group_size)
            group_path = context_root / GROUP_FOLDER_NAMES[group_size] / group_name
            if not group_path.is_dir():
                missing_group = True
                break
            group_dirs[group_size] = str(group_path.resolve())

        if not missing_group:
            candidates.append(
                SampleCandidate(
                    dataset=dataset_name,
                    dialogue_id=dialogue_id,
                    end_utterance_id=end_utterance_id,
                    anchor_name=sample_dir.name,
                    group_dirs=group_dirs,
                )
            )

    return candidates


def choose_samples(
    rng: random.Random,
    candidates: list[SampleCandidate],
    sample_count: int,
) -> list[SampleCandidate]:
    if len(candidates) < sample_count:
        raise ValueError(
            f"Requested {sample_count} samples but found only {len(candidates)} valid candidates."
        )
    return rng.sample(candidates, sample_count)


def reset_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def copy_sample_groups(
    dataset_output_root: Path,
    batch_name: str,
    candidate: SampleCandidate,
) -> dict[int, str]:
    copied_dirs: dict[int, str] = {}
    batch_root = dataset_output_root / batch_name

    for group_size, source_dir in candidate.group_dirs.items():
        target_dir = batch_root / GROUP_FOLDER_NAMES[group_size] / Path(source_dir).name
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_dir, target_dir)
        copied_dirs[group_size] = str(target_dir.resolve())

    return copied_dirs


def zip_batch_folder(batch_root: Path) -> Path:
    archive_base = batch_root.parent / batch_root.name
    zip_path = Path(
        shutil.make_archive(
            base_name=str(archive_base),
            format="zip",
            root_dir=str(batch_root.parent),
            base_dir=batch_root.name,
        )
    )
    return zip_path.resolve()


def write_summary(output_root: Path, summary: dict) -> None:
    json_path = output_root / "selection_summary.json"
    txt_path = output_root / "selection_summary.txt"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "Human Eval Sample Selection",
        "===========================",
        "",
        f"Data root: {summary['data_root']}",
        f"Output root: {summary['output_root']}",
        f"Seed: {summary['seed']}",
        "",
    ]

    for dataset_summary in summary["datasets"]:
        lines.extend(
            [
                dataset_summary["dataset"],
                "-" * len(dataset_summary["dataset"]),
                f"Available valid candidates: {dataset_summary['available_candidate_count']}",
                f"Selected samples: {dataset_summary['selected_sample_count']}",
                f"Batch count: {dataset_summary['batch_count']}",
                f"Batch size: {dataset_summary['batch_size']}",
                f"Group sizes copied: {dataset_summary['group_sizes']}",
                "",
            ]
        )

        for batch in dataset_summary["batches"]:
            lines.append(
                f"{batch['batch_name']}: {len(batch['samples'])} samples, zip={batch['zip_path']}"
            )
            lines.append(
                "  anchors: "
                + ", ".join(sample["anchor_name"] for sample in batch["samples"])
            )
        lines.append("")

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset_summary(
    dataset_name: str,
    candidates: list[SampleCandidate],
    selected_candidates: list[SampleCandidate],
    batch_count: int,
    group_sizes: tuple[int, ...],
    dataset_output_root: Path,
) -> dict:
    if len(selected_candidates) % batch_count != 0:
        raise ValueError(
            f"Selected sample count {len(selected_candidates)} is not divisible by batch_count {batch_count}."
        )

    batch_size = len(selected_candidates) // batch_count
    batches: list[dict] = []

    for batch_index in range(batch_count):
        batch_name = f"batch{batch_index + 1:02d}"
        batch_root = dataset_output_root / batch_name
        batch_candidates = selected_candidates[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ]

        sample_entries: list[dict] = []
        for candidate in batch_candidates:
            copied_dirs = copy_sample_groups(
                dataset_output_root=dataset_output_root,
                batch_name=batch_name,
                candidate=candidate,
            )
            sample_entries.append(
                {
                    **asdict(candidate),
                    "copied_dirs": copied_dirs,
                }
            )

        zip_path = zip_batch_folder(batch_root)
        batches.append(
            {
                "batch_name": batch_name,
                "batch_root": str(batch_root.resolve()),
                "zip_path": str(zip_path),
                "samples": sample_entries,
            }
        )

    return {
        "dataset": dataset_name,
        "available_candidate_count": len(candidates),
        "selected_sample_count": len(selected_candidates),
        "batch_count": batch_count,
        "batch_size": batch_size,
        "group_sizes": list(group_sizes),
        "dataset_output_root": str(dataset_output_root.resolve()),
        "batches": batches,
    }


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    reset_output_root(output_root=output_root, overwrite=args.overwrite)

    rng = random.Random(args.seed)
    dataset_summaries: list[dict] = []

    for dataset_name, config in DATASET_CONFIGS.items():
        dataset_root = data_root / dataset_name
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

        candidates = discover_candidates(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            group_sizes=config["group_sizes"],
        )
        selected_candidates = choose_samples(
            rng=rng,
            candidates=candidates,
            sample_count=config["sample_count"],
        )

        dataset_output_root = output_root / dataset_name
        dataset_output_root.mkdir(parents=True, exist_ok=True)
        dataset_summary = build_dataset_summary(
            dataset_name=dataset_name,
            candidates=candidates,
            selected_candidates=selected_candidates,
            batch_count=config["batch_count"],
            group_sizes=config["group_sizes"],
            dataset_output_root=dataset_output_root,
        )
        dataset_summaries.append(dataset_summary)

        print(
            f"[INFO] {dataset_name}: selected {dataset_summary['selected_sample_count']} "
            f"samples from {dataset_summary['available_candidate_count']} candidates."
        )

    summary = {
        "data_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "datasets": dataset_summaries,
    }
    write_summary(output_root=output_root, summary=summary)

    print(f"[INFO] Wrote outputs to {output_root}")


if __name__ == "__main__":
    main()
