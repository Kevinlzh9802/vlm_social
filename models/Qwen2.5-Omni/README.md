# Qwen2.5-Omni Inference Wrapper

This directory contains the local Qwen2.5-Omni inference code and Slurm job scripts used by this repository.

Original upstream project: https://github.com/QwenLM/Qwen2.5-Omni

For model details, licenses, checkpoints, and general examples, refer to the upstream repository and the Qwen model pages. The notes below describe how the repository-specific batch inference scripts are wired.

## Required Paths

Set these paths before submitting jobs:

```bash
export QWEN_PROJECT_ROOT=/path/to/GesBench/models/Qwen2.5-Omni
export DATA_ROOT=/path/to/data/gestalt_bench
export MODEL_ROOT=/path/to/models
export APPTAINER_ROOT=/path/to/apptainers
```

Optional:

```bash
export HF_CACHE=/path/to/huggingface-cache
export LOG_DIR=logs/qwen2.5-omni
```

Expected files and directories:

- Apptainer image: `${APPTAINER_ROOT}/qwen2.5-omni-inference.sif`
- Model checkpoints: `${MODEL_ROOT}/Qwen2.5-Omni-7B` or `${MODEL_ROOT}/Qwen2.5-Omni-3B`
- Prompt config: `${QWEN_PROJECT_ROOT}/prompts/prompts.json`
- Batch data under `${DATA_ROOT}`

Create the Slurm log directory before submitting:

```bash
mkdir -p logs/qwen2.5-omni
```

## Job Scripts

`job_scripts/inference_test.sh`

- Runs a quick smoke test through `test_infer.py`.
- Optional model override:

```bash
sbatch job_scripts/inference_test.sh --model "${MODEL_ROOT}/Qwen2.5-Omni-7B"
```

`job_scripts/inference_batch.sh`

- Main <cluster2>-style batch inference script.
- Supports `context` and `nested` modes.
- Supports model sizes `7B` and `3B`.
- Supports `single-turn` and `multi-turn`; nested mode forces `multi-turn`.
- Supports annotator mode with `-annotator <n>`, which runs utterance groups 1, 2, and 3.

Examples:

```bash
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --utt 2 --batch 2 --model 7B --prompt affordance
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode nested --utt 3 --batch 12 --model 3B --prompt intention
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --batch 2 --model 7B --prompt intention -annotator 1
```

`job_scripts/inference_batch_<cluster1>.sh`

- <cluster1>-style batch inference script.
- Supports the same standard `context` and `nested` modes as `inference_batch.sh`.
- Supports annotated task-2 runs with `--annotated`.
- With `--annotated`, `--comparison` switches to comparison data/results and `--no-audio` omits separate audio inputs.

Examples:

```bash
sbatch job_scripts/inference_batch_<cluster1>.sh --dataset mintrec2 --mode context --utt 2 --batch 2 --model 7B --prompt affordance
sbatch job_scripts/inference_batch_<cluster1>.sh --dataset mintrec2 --mode context --batch 2 --model 7B --prompt intention --annotated
sbatch job_scripts/inference_batch_<cluster1>.sh --dataset mintrec2 --mode context --batch 2 --model 7B --prompt intention --annotated --comparison
sbatch job_scripts/inference_batch_<cluster1>.sh --dataset mintrec2 --mode context --batch 2 --model 7B --prompt intention --annotated --no-audio
```

## Data and Output Layout

For standard context runs, the scripts read:

```text
${DATA_ROOT}/${dataset}/context/${utt}-utt_group/batch${batch_id}
```

For nested runs, the scripts read:

```text
${DATA_ROOT}/${dataset}/nested/data/batch${batch_id}
```

Standard outputs are written to:

```text
${DATA_ROOT}/results/qwen2.5/${dataset}/${mode}/${utt}-utt_group/${model_size}_${prompt}_${conversation_mode}/batch${batch_id}.json
```

<cluster1> annotated outputs are written under one of:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/results/qwen2.5
${DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison/qwen2.5
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio/qwen2.5
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison/qwen2.5
```

## Notes

- Submit from this directory or set `QWEN_PROJECT_ROOT` explicitly.
- The scripts bind `QWEN_PROJECT_ROOT` to `/workspace` in the container.
- `HF_CACHE` is mounted to `/opt/huggingface`.
- The prompt choice must exist in `prompts/prompts.json`; current scripts expect choices such as `intention` and `affordance`.
