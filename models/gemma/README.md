# Gemma Inference Wrapper

This directory contains the repository-specific Gemma multimodal batch inference wrapper. The runner loads a local or Hugging Face Gemma checkpoint with Transformers and applies the Gestalt Bench prompt and data layout.

Main file:

- `batch_infer_context.py`: scans batch folders for video/audio inputs, runs Gemma inference, and writes grouped JSON outputs.

The Slurm wrappers are in the repository root:

- `job_scripts/gemma_<cluster1>.sh`
- `job_scripts/gemma_<cluster2>.sh`
- `job_scripts/gemma_<cluster1>_old.sh`

## Required Paths

Set these paths before submitting jobs:

```bash
export PROJECT_ROOT=/path/to/gesbench
export DATA_ROOT=/path/to/data/gestalt_bench
export MODEL_ROOT=/path/to/models
export APPTAINER_ROOT=/path/to/apptainers
```

Optional:

```bash
export HF_CACHE=/path/to/huggingface-cache
```

Expected files and directories:

- Apptainer image: `${APPTAINER_ROOT}/gemma.sif`
- Default model checkpoint: `${MODEL_ROOT}/GemmaE4B`
- Prompt config: `models/gemini/configs/prompts.json`
- Batch data under `${DATA_ROOT}`

The model path can also be overridden per job with `--model-path`.

Create the Slurm log directory before submitting from the repository root:

```bash
mkdir -p logs/gemma
```

## Batch Inference

Standard context run:

```bash
sbatch job_scripts/gemma_<cluster1>.sh --dataset mintrec2 --utt 1 --batch 1 --prompt intention
sbatch job_scripts/gemma_<cluster1>.sh --dataset mintrec2 --utt 2 --batch 3 --prompt affordance --conversation-mode multi-turn
```

Annotated task-2 run:

```bash
sbatch job_scripts/gemma_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated
sbatch job_scripts/gemma_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --comparison
sbatch job_scripts/gemma_<cluster1>.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --no-audio
```

Useful runtime options:

```bash
--conversation-mode single-turn|multi-turn
--model-path /path/to/GemmaE4B
--sif-path /path/to/gemma.sif
--data-root /path/to/data/gestalt_bench
--max-new-tokens 512
--max-video-frames 32
--enable-thinking
--do-sample
--no-audio
```

## Data and Output Layout

The Gemma runner expects the selected `--data-root` to contain immediate sample subfolders:

```text
batch01/
  d1u3/
    d1u3_clip1.mp4
    d1u3_clip1.wav
  d2u8/
    d2u8_clip1.mp4
    d2u8_clip1.wav
```

In normal Slurm usage, the wrapper passes:

```text
${DATA_ROOT}/${dataset}/context/${utt}-utt_group/batch${batch_id}
```

Standard outputs are written to:

```text
${DATA_ROOT}/results/gemma-4-e4b/${dataset}/context/${utt}-utt_group/E4B_${prompt}_${conversation_mode}/batch${batch_id}.json
```

Annotated task-2 input roots are:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/data
${DATA_ROOT}/human_eval/task2/manipulation_full/data_comparison
```

Annotated task-2 outputs are written under one of:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/results/gemma-4-e4b
${DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison/gemma-4-e4b
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio/gemma-4-e4b
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison/gemma-4-e4b
```

## Prompt and Conversation Modes

Prompt choices are resolved from `models/gemini/configs/prompts.json`. The job scripts validate that the selected prompt exists for the requested utterance count and conversation mode.

- `single-turn`: each clip is inferred independently with the selected prompt.
- `multi-turn`: each sample subfolder is treated as one conversation, using the first-turn prompt for the first clip and the follow-up prompt for later clips.

With `--no-audio`, the runner omits separate `.wav` files and uses video-only inputs.

