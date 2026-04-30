# Gestalt Bench

This repository contains data preparation, multimodal model inference, eye-tracking extraction, human annotation extraction, and analysis code for Gestalt Bench experiments.

The scripts are designed to run either directly or through Slurm job wrappers in `job_scripts/`. Most job scripts use environment-variable based paths so the repository can be moved between systems without editing hardcoded local paths.

## Repository Layout

- `dataset/`: dataset preparation utilities, dialogue partitioning, and human-evaluation sample selection.
- `models/`: model-specific inference wrappers and job scripts.
- `eyetrack/`: Pupil Core gaze extraction, focus plotting, and gaze-corrupted data generation utilities.
- `experiments/analysis/`: semantic-similarity analysis, human/model comparisons, and plot generation.
- `tasks/`: task-level batch helpers.
- `apptainer/`: container definitions used by the Slurm jobs.
- `job_scripts/`: root Slurm wrappers for dataset processing, inference, eye tracking, and analysis.

## Common Paths

Set paths before running Slurm jobs:

```bash
export PROJECT_ROOT=/path/to/gesbench
export DATA_ROOT=/path/to/data/gestalt_bench
export RESULTS_ROOT=/path/to/results/gestalt_bench
export MODEL_ROOT=/path/to/models
export APPTAINER_ROOT=/path/to/apptainers
```

Some workflows also require:

```bash
export HF_CACHE=/path/to/huggingface-cache
export API_KEY_FILE=/path/to/api_key.txt
```

Submit jobs from the repository root and create `logs/` first:

```bash
mkdir -p logs
```

## Data Preparation

Dataset utilities create the clip structure used by the inference scripts. The main steps are:

1. Partition raw dialogue videos into aligned `1-utt`, `2-utt`, and `3-utt` context groups.
2. Select and zip human-evaluation samples from the partitioned context folders.
3. Optionally summarize prepared datasets and result folders.

See [dataset/README.md](dataset/README.md) for the dialogue partition algorithm, output layout, sample selection policy, and command examples.

## Model Inference

The repository includes wrappers for multiple multimodal models:

- Qwen2.5-Omni: [models/Qwen2.5-Omni/README.md](models/Qwen2.5-Omni/README.md)
- Ming-Lite-Omni: [models/Ming-lite-omni/README.md](models/Ming-lite-omni/README.md)
- Gemini Batch API: [models/gemini/README.md](models/gemini/README.md)
- Gemma: [models/gemma/README.md](models/gemma/README.md)

These wrappers read the prepared context folders, run single-turn or multi-turn inference where supported, and write JSON results under the configured result roots. Gemini uses a submit/retrieve workflow through the Batch API; local models run inside Apptainer images.

## Eye Tracking

Eye-tracking scripts extract Pupil Core gaze data for videos shown in the human annotation interface. The workflow aligns annotation video start/end timestamps with Pupil time, filters gaze samples by confidence, maps screen gaze points into video coordinates, and writes focus plots plus extraction summaries.

The same gaze alignment can also generate task-2 manipulated video data by blurring or blocking the gaze-centered region.

See [eyetrack/README.md](eyetrack/README.md) for the extraction pipeline and gaze-corrupted data workflow.

## Human Annotation Extraction

Human annotation JSONs can be normalized into linked CSV/JSON tables before analysis. The extraction step maps annotation rows to task media, keeps annotator/task metadata, and produces reusable files such as `human_annotations.csv` and `human_annotations_linked.json`.

The relevant wrapper is:

```bash
sbatch job_scripts/parse_human_annotations_<cluster1>.sh
```

Human annotation extraction is also invoked automatically by the human annotation and human/model similarity analyses when raw annotation directories are provided. See [experiments/analysis/README.md](experiments/analysis/README.md) for details.

## Analysis and Plots

Analysis scripts compute SentenceTransformer cosine similarities over model outputs and human annotations. The main analyses are:

- Model partial-to-final similarity across cumulative clips.
- Human partial-to-full annotation similarity.
- Human-vs-model answer similarity.
- Task-2 manipulated-result similarity against full benchmark answers.
- Semantic turnover metrics from neighboring clip similarities.

Plots and tables are generated under the configured data/result roots, with optional cached plot data for regenerating aggregate figures without re-embedding text.

See [experiments/analysis/README.md](experiments/analysis/README.md) for the metrics, plot outputs, and Slurm commands.

## Typical Workflow

1. Prepare dialogue partitions and human-evaluation samples with the dataset scripts.
2. Run model inference for the selected models and prompts.
3. Extract human annotations and, when needed, eye-tracking focus data.
4. Generate gaze-corrupted task-2 data and run inference on those manipulated clips.
5. Run analysis scripts to compute similarities, semantic turnover metrics, tables, and plots.
