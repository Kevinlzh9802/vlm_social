# Eyetrack Extraction Copy From 64b3e28

This folder contains a copied subset of the repository at commit `64b3e28`:

- `eyetrack/annotation_intervals.py`
- `eyetrack/eyetrack_annotation.py`
- `eyetrack/focus_plot.py`
- `eyetrack/gaze_extraction.py`
- `eyetrack/eyetrack_test.py`
- `job_scripts/eyetrack_annotation_delftblue.sh`

Use this to reproduce the old `extraction_focus` behavior without checking out
the whole repository.

DAIC submission:

```bash
sbatch legacy_64b3e28/job_scripts/eyetrack_annotation_daic.sh
```

Default output:

```text
/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/human_eval/task2/extraction_focus_64b3e28
```

The DAIC wrapper binds `legacy_64b3e28` as `/workspace`, so the script runs the
copied historical files instead of the current repo files.
