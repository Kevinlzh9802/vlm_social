# Analysis Experiments

This directory contains the scripts used to compute semantic similarity metrics and generate plots from model outputs and human annotations.

## Required Paths

Set these paths before using the Slurm wrappers:

```bash
export PROJECT_ROOT=/path/to/gesbench
export DATA_ROOT=/path/to/data/gestalt_bench
export RESULTS_ROOT=/path/to/results/gestalt_bench
export APPTAINER_ROOT=/path/to/apptainers
```

Expected image:

```text
${APPTAINER_ROOT}/analysis.sif
```

Most scripts use a SentenceTransformer model. By default this is `all-MiniLM-L6-v2`; on compute nodes, passing a local `--model-path` is usually preferable.

## Similarity Metrics

All semantic similarities are cosine similarities between SentenceTransformer embeddings.

The shared metric logic is in `metrics.py`:

- Invalid outputs are skipped when they start with `[ERROR]` or match CUDA out-of-memory patterns.
- For each ordered clip sequence, responses are embedded in clip order.
- `clip_to_final_similarities`: cosine similarity between every clip response and the final clip response.
- `neighboring_similarities`: cosine similarity between adjacent clip responses.
- Semantic turnover counts an adjacent pair as a turnover when neighboring similarity is below a threshold.
- Semantic turnover ratio is `turnover_count / clip_count`.
- Weighted average semantic-turnover position weights each adjacent position by `1 - neighboring_similarity` and normalizes by clip count.

Clip progress is binned by `position / clip_count`, rounded into `--progress-partitions` bins. The default is 20 bins.

## Model Partial-to-Final Analysis

Run:

```bash
sbatch job_scripts/analysis_<cluster1>.sh
```

The wrapper calls `main.py`. It reads model result JSON files under:

```text
${DATA_ROOT}/results
${RESULTS_ROOT}/human_eval/gemini
${DATA_ROOT}/results/gemma-4-e4b
```

Model result folders are discovered by dataset structure (`mintrec2`, `meld`, `seamless_interaction`) and `batch*.json` files. The parser groups entries by dialogue or clip prefix, orders clips by clip index, extracts assistant text from `assistant`, `response`, or `response_text`, and computes the metrics above.

Useful options:

```bash
--skip-gemini
--skip-gemma
--skip-human-overlay
--save-plot-data
--from-plot-data /path/to/analysis_plot_data.json
--turnover-thresholds 0.3 0.5 0.7 0.9
--progress-partitions 20
--with-scatter
--model-path /path/to/sentence-transformer
```

Primary outputs are written under `${DATA_ROOT}/plots`:

- Per-result-folder plots:
  - `clip_to_final_similarity_with_scatter.png`
  - `clip_to_final_similarity_percentiles_only.png`
  - `clip_to_final_similarity_mean_only.png`
  - `neighbor_similarity_by_clip_count.png`
  - `semantic_turnover_by_clip_count_t*.png`
- Combined plots across models for each utterance count and task:
  - `combined_clip_to_final_percentile_p*_...png`
  - `combined_clip_to_final_mean_...png`
  - `combined_st_threshold_...png`
  - human/model overlay plots when human annotation summaries are provided
- Tables:
  - `wastp_summary.csv`
  - `wastp_summary.md`

With `--save-plot-data`, reusable numeric data are written to:

```text
${DATA_ROOT}/plots/plot_data/analysis_plot_data.json
${DATA_ROOT}/plots/plot_data/analysis_plot_points.csv
${DATA_ROOT}/plots/plot_data/analysis_plot_bins.csv
```

Use `--from-plot-data` to regenerate aggregate plots without re-reading result JSON files or re-embedding text.

## Human Annotation Partial-to-Full Analysis

Run:

```bash
sbatch job_scripts/human_annotation_similarity_<cluster1>.sh
```

This wrapper calls `human_annotation_similarity.py`. The script accepts either a pre-extracted `human_annotations.csv` or a directory of raw annotation JSONs. In normal usage it reads:

```text
${DATA_ROOT}/human_eval/task1/annotations
${DATA_ROOT}/human_eval/task1/task1.json
```

It extracts human annotation text, links rows to task media, groups clips by dataset, task, utterance count, annotator, and sample, and compares each partial clip annotation with the full/final clip annotation.

Outputs:

- Extracted annotations:
  - `human_annotations.csv`
  - `human_annotations_linked.json`
- Plot data:
  - `partial_to_full_percentiles.csv`
  - `partial_to_full_percentiles.json`
  - `partial_to_full_points.csv`
  - `partial_to_full_points.json`
- Plots by prompt:
  - per-dataset/utterance partial-to-full percentile and average plots
  - all-dataset aggregate plots

The `partial_to_full_points.csv` file includes `neighboring_similarity_to_next`, which allows `analysis_<cluster1>.sh` to overlay human semantic-turnover curves on the model plots.

## Human-vs-Model Similarity

Run:

```bash
sbatch job_scripts/human_model_similarity_<cluster1>.sh
```

This wrapper calls `human_model_similarity.py`. It extracts or loads human annotations, indexes model results from the configured result roots, aligns human and model clips by dataset, task, utterance count, and clip identifier, then computes diagonal cosine similarity between paired human/model embeddings.

Outputs:

- `human_model_similarity_summary.csv`
- `human_model_similarity_summary.json`
- `human_model_similarity_points.csv`
- `human_model_similarity_points.json`
- multi-model plots for mean, median, p25, p50, p75, and relative-to-final variants

## Task-2 Manipulation Similarity

Run:

```bash
sbatch job_scripts/manipulation_result_similarity_<cluster1>.sh
```

This wrapper calls `manipulation_result_similarity.py`. It compares model answers on gaze-manipulated task-2 data against the corresponding full benchmark answers.

Inputs:

```text
source:    ${DATA_ROOT}/human_eval/task2/manipulation_full/results
reference: ${DATA_ROOT}/results
```

Modes:

```bash
--comparison
--no-audio
```

For each matching model, prompt, dataset, utterance count, and clip id, it embeds the source and reference answers and computes cosine similarity. It writes point-level comparison data plus summaries by dataset, utterance count, and overall weighted means.

Outputs include:

- `manipulation_result_similarity_points.csv`
- `manipulation_result_similarity_by_dataset_utt.csv`
- `manipulation_result_similarity_by_utt.csv`
- `manipulation_result_similarity_by_dataset.csv`
- `manipulation_result_similarity_overall_weighted.csv`
- `manipulation_result_similarity_table.csv`
- `manipulation_result_similarity_table.md`

## Parsing Human Annotations Only

Run:

```bash
sbatch job_scripts/parse_human_annotations_<cluster1>.sh
```

This calls `parse_human_annotations.py` and writes linked human annotation CSV/JSON files without computing embeddings or plots. Use it when you only need the normalized annotation table for inspection or downstream scripts.

