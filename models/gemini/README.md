# Gemini Inference Wrapper

This directory contains the repository-specific Gemini API wrappers used for batch inference. It is not a copy of a model repository; it wraps the Google Gemini Batch API around the Gestalt Bench video/audio folder layout.

Main files:

- `gemini.py`: shared Gemini client, API-key loading, single-video helper, and prompt resolution.
- `gemini_batch.py`: submit-only Batch API runner. It uploads media, writes a request JSONL file, creates a Gemini batch job, and records the job in a registry.
- `gemini_retrieve.py`: registry-driven retrieval runner. It checks pending jobs, downloads completed outputs, writes result JSON files, and updates registry statuses.
- `gemini_list_files.py`: utility for listing or deleting uploaded Gemini Files API files.
- `prompt_utils.py`: shared prompt-config helpers.

## Required Paths

For Slurm jobs, set:

```bash
export DATA_ROOT=/path/to/data/gestalt_bench
export RESULTS_ROOT=/path/to/results/gestalt_bench
export APPTAINER_ROOT=/path/to/apptainers
export API_KEY_FILE=/path/to/api_key.txt
```

Expected files:

- Apptainer image: `${APPTAINER_ROOT}/gemini.sif`
- Gemini API key file: `${API_KEY_FILE}`
- Prompt config: `models/gemini/configs/prompts.json`
- Batch data under `${DATA_ROOT}`

Create the Slurm log directory before submitting from the repository root:

```bash
mkdir -p logs/gemini-batch
```

## Batch Workflow

Gemini batch inference is a two-step process:

1. Submit one or more batch jobs with `job_scripts/gemini_batch_<cluster1>.sh`.
2. Retrieve finished jobs later with `job_scripts/gemini_retrieve_<cluster1>.sh`.

Standard human-evaluation sample submission:

```bash
sbatch job_scripts/gemini_batch_<cluster1>.sh --dataset mintrec2 --utt 1 --batch 1 --prompt intention
sbatch job_scripts/gemini_batch_<cluster1>.sh --dataset mintrec2 --utt 2 --batch 3 --prompt affordance --gemini-mode 2.5-flash
```

Retrieve completed standard jobs:

```bash
sbatch job_scripts/gemini_retrieve_<cluster1>.sh
```

Annotated task-2 submission:

```bash
sbatch job_scripts/gemini_batch_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated
sbatch job_scripts/gemini_batch_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --comparison
sbatch job_scripts/gemini_batch_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --no-audio
```

Retrieve annotated task-2 jobs with the same mode flags:

```bash
sbatch job_scripts/gemini_retrieve_<cluster1>.sh --annotated
sbatch job_scripts/gemini_retrieve_<cluster1>.sh --annotated --comparison
sbatch job_scripts/gemini_retrieve_<cluster1>.sh --annotated --no-audio
```

## Data and Output Layout

The submit script reads standard batches from:

```text
${DATA_ROOT}/human_eval/samples/${dataset}/context/${utt}-utt_group/batch${batch_id}
```

Each batch folder must contain immediate sample subfolders. Each sample subfolder contains `.mp4` files and, unless `--no-audio` is used, matching same-stem `.wav` files.

Standard Gemini outputs are written to:

```text
${RESULTS_ROOT}/human_eval/gemini/${dataset}/context/${utt}-utt_group/${gemini_mode}_${prompt}_single-turn/batch${batch_id}.json
```

Annotated task-2 input roots are:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/data
${DATA_ROOT}/human_eval/task2/manipulation_full/data_comparison
```

Annotated task-2 outputs are written under one of:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/results/gemini
${DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison/gemini
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio/gemini
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison/gemini
```

## Registry and Uploaded Files

`gemini_batch.py` appends every submitted job to `gemini_registry.json` under the selected output root. The registry stores the job name, dataset, prompt, utterance count, target output JSON path, and the media map needed by retrieval.

`gemini_retrieve.py` is safe to rerun. It skips already retrieved jobs and only writes results for jobs that have reached a terminal state.

To inspect or clean uploaded Gemini Files API items:

```bash
sbatch job_scripts/gemini_list_files_<cluster1>.sh
```

