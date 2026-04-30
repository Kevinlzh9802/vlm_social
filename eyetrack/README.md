# Eyetrack Extraction

This experiment extracts gaze data from Pupil Core recordings for the videos shown during the human annotation tasks. The implementation lives in `eyetrack/`; this directory documents the experiment workflow.

## Required Paths

Set these paths before using the Slurm wrappers:

```bash
export PROJECT_ROOT=/path/to/gesbench
export DATA_ROOT=/path/to/data/gestalt_bench
export APPTAINER_ROOT=/path/to/apptainers
```

Expected inputs:

- Pupil recording parent: `${DATA_ROOT}/human_eval/pupil`
- Annotation JSONs: `${DATA_ROOT}/human_eval/task2/results`
- Task video list: `${DATA_ROOT}/human_eval/task2/task2.json`
- Local video root: `${DATA_ROOT}/human_eval/videos`
- Apptainer image: `${APPTAINER_ROOT}/eyetrack.sif`

Pupil recording folders are expected to be named:

```text
T<task_number>_<task_instance_id>_annotator<annotator_number>
```

For example:

```text
T2_7_annotator1/
T2_7_annotator2/
```

Each recording should contain `info.player.json` and a Pupil surface export under:

```text
exports/<three_digit_export_id>/surfaces/gaze_positions_on_surface_*.csv
```

## Focus Plot Extraction

Run:

```bash
sbatch job_scripts/eyetrack_annotation_<cluster1>.sh
```

or, on <cluster2>:

```bash
sbatch job_scripts/eyetrack_annotation_<cluster2>.sh
```

The wrapper calls:

```bash
python eyetrack/eyetrack_annotation.py \
  <pupil_parent> \
  <annotation_dir> \
  <local_path_prefix> \
  --video-json <task2.json> \
  --output-dir <output_dir> \
  --timing-csv <output_dir>/timing_tables.csv \
  --summary-csv <output_dir>/extraction_summary.csv
```

The extraction pipeline does the following:

1. Finds annotation files named `T{x}_{y}.json`.
2. Reads annotator journeys from each JSON and selects a response with `--response-selection latest-submitted` by default.
3. Extracts each video's `video_start_time`, `video_end_time`, media paths, and frontend timing metadata.
4. Matches each annotation file and annotator to the corresponding Pupil recording folder.
5. Loads the latest Pupil surface gaze CSV. Rows are read from `world_timestamp`, `x_norm`, `y_norm`, `confidence`, and optional `on_surf`.
6. Converts annotation UNIX/system timestamps to Pupil time with `info.player.json`:

```text
Pupil Time = system_time + (start_time_synced_s - start_time_system_s)
```

7. Keeps gaze samples whose timestamps fall inside the annotation video interval and whose confidence is at least `--confidence` (default `0.6`).
8. Resolves the viewed video path from `task2.json` and the local media prefix.
9. Maps normalized screen gaze coordinates to normalized video coordinates.
10. Writes one static focus plot per annotation video plus CSV summaries.

## Gaze Mapping

Two screen-to-video mappings are supported:

- `measured-player`: uses the measured browser/player geometry in `eyetrack/focus_plot.py`. This is the default.
- `legacy-extraction`: reproduces the older square-region mapping with `--video-screen-ratio` defaulting to `0.7`.

Use:

```bash
sbatch job_scripts/eyetrack_annotation_<cluster1>.sh --focus-mapping legacy-extraction
```

## Outputs

The default focus extraction output is:

```text
${DATA_ROOT}/human_eval/task2/extraction_focus_new
```

Important files:

- `timing_tables.csv`: one row per annotation video interval.
- `extraction_summary.csv`: one row per generated focus plot, including recording paths, gaze source paths, sample counts, mapping mode, and mean mapped gaze coordinates.
- `T{x}_{y}_annotator{n}/*.png`: focus plots with mapped gaze points over a representative video frame.

## Gaze-Corrupted Task-2 Data

The gaze-blocking workflow uses the same extracted gaze alignment to create manipulated video clips:

```bash
sbatch job_scripts/gaze_blocked_partition_<cluster1>.sh
```

It calls `eyetrack/gaze_blocked_partition.py`, builds gaze points from the annotation intervals, maps them into video coordinates, and writes task-2 `context/{1,2,3}-utt_group` data with a gaze-centered blur or block mask.

Useful options:

```bash
--effect blur|block
--focus-region-ratio 0.18
--focus-region-shape circle|square
--max-gaze-gap 0.5
--gaze-mapping measured-player|legacy-extraction
--comparison
--utt 1,2,3
```

`--comparison` shifts the mask center away from the gaze point and writes comparison data. The generated manipulated data are used by the task-2 model inference scripts and later compared against full benchmark outputs by `experiments/analysis/manipulation_result_similarity.py`.

## Debug Utilities

- `eyetrack/annotation_intervals.py`: prints video timing tables from annotation JSON files.
- `eyetrack/eyetrack_test.py`: extracts raw `gaze.pldata` samples for manually supplied intervals.
- `eyetrack/pupil_world_gaze_overlay.py`: writes `world_gaze.mp4` and `world_gaze_points.csv` inside Pupil recording folders.
- `eyetrack/gaze_extraction.py`: shared loading, timestamp alignment, media resolution, and interval filtering helpers.
- `eyetrack/focus_plot.py`: screen-to-video mapping and static focus plotting.

