# Ming-Lite-Omni Inference Wrapper

This directory contains the local Ming-Lite-Omni inference code and Slurm job scripts used by this repository.

Original upstream project: https://github.com/inclusionAI/Ming

For model details, licenses, checkpoints, and general examples, refer to the upstream repository and the official model pages. The notes below describe how the repository-specific batch inference scripts are wired.

## Required Paths

Set these paths before submitting jobs:

```bash
export MING_PROJECT_ROOT=/path/to/vlm_social/models/Ming-lite-omni
export DATA_ROOT=/path/to/data/gestalt_bench
export APPTAINER_ROOT=/path/to/apptainers
```

Optional:

```bash
export LOG_DIR=logs/ming-lite-omni
export SIF_PATH=/path/to/ming-lite-omni.sif
export APPTAINER_TMPDIR=/path/to/tmp
```

Expected files and directories:

- Apptainer image: `${APPTAINER_ROOT}/ming-lite-omni.sif` unless `SIF_PATH` is set
- Prompt config: `${MING_PROJECT_ROOT}/prompts/prompts.json`
- Inference script: `${MING_PROJECT_ROOT}/batch_infer.py`
- Batch data under `${DATA_ROOT}`

Create the Slurm log directory before submitting:

```bash
mkdir -p logs/ming-lite-omni
```

## Job Scripts

`job_scripts/build_apptainer_daic.sh`

- Builds the Ming-Lite-Omni Apptainer image from `apptainer/Ming-lite-omni.def`.
- Writes to `${SIF_PATH}` when set, otherwise `${APPTAINER_ROOT}/ming-lite-omni.sif`.

Example:

```bash
sbatch --export=ALL,MING_PROJECT_ROOT=/path/to/Ming-lite-omni,APPTAINER_ROOT=/path/to/apptainers job_scripts/build_apptainer_daic.sh
```

`job_scripts/inference_test.sh`

- Runs a selected local test script inside the Apptainer image.
- Defaults to `test_infer.py`.

Examples:

```bash
sbatch job_scripts/inference_test.sh
sbatch job_scripts/inference_test.sh test_infer_gen_image.py
sbatch job_scripts/inference_test.sh test_audio_tasks.py
```

`job_scripts/inference_batch.sh`

- Main DelftBlue-style batch inference script.
- Supports `context` and `nested` modes.
- Supports `single-turn` and `multi-turn`; nested mode forces `multi-turn`.
- Supports annotator mode with `--annotator <n>`, which runs utterance groups 1, 2, and 3.
- Supports `--attn-implementation auto|eager|sdpa|flash_attention_2`.

Examples:

```bash
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --utt 2 --batch 2 --prompt affordance
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode nested --utt 3 --batch 12 --prompt intention
sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --batch 2 --prompt intention --annotator 1
```

`job_scripts/inference_batch_daic.sh`

- DAIC-style batch inference script.
- Supports `--annotated` mode for task-2 manipulation data.
- With `--annotated`, `--comparison` switches to comparison data/results and `--no-audio` omits separate audio inputs.
- Supports `--max-frames <n>` and `--attn-implementation auto|eager|sdpa|flash_attention_2`.

Examples:

```bash
sbatch job_scripts/inference_batch_daic.sh --dataset mintrec2 --mode context --utt 2 --batch 2 --prompt affordance
sbatch job_scripts/inference_batch_daic.sh --dataset mintrec2 --mode context --batch 2 --prompt intention --annotated
sbatch job_scripts/inference_batch_daic.sh --dataset mintrec2 --mode context --batch 2 --prompt intention --annotated --comparison
sbatch job_scripts/inference_batch_daic.sh --dataset mintrec2 --mode context --batch 2 --prompt intention --annotated --no-audio
```

`job_scripts/inference_batch_daic_a40.sh` and `job_scripts/inference_batch_daic_old.sh`

- Alternative DAIC variants kept for compatibility with earlier A40/legacy runs.
- They use the same path variables and data layout as the main Ming batch scripts.

`job_scripts/inference_qwen.sh`

- Qwen2.5-Omni batch wrapper kept in this directory for convenience.
- It uses Qwen-specific variables: `QWEN_PROJECT_ROOT`, `MODEL_ROOT`, `DATA_ROOT`, `APPTAINER_ROOT`, and `HF_CACHE`.

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
${DATA_ROOT}/results/ming-lite-omni/${dataset}/${mode}/${utt}-utt_group/Ming-lite-omni_${prompt}_${conversation_mode}/batch${batch_id}.json
```

DAIC annotated outputs are written under one of:

```text
${DATA_ROOT}/human_eval/task2/manipulation_full/results/ming-lite-omni
${DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison/ming-lite-omni
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio/ming-lite-omni
${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison/ming-lite-omni
```

## Notes

- Submit from this directory or set `MING_PROJECT_ROOT` explicitly.
- The scripts bind `MING_PROJECT_ROOT` to `/workspace` in the container.
- The Ming scripts keep the local Triton/libcuda workaround used by the original inference jobs.
- The prompt choice must exist in `prompts/prompts.json`; current scripts expect choices such as `intention` and `affordance`.
