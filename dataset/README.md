# Dataset Utilities

This directory contains the data preparation utilities used by the repository. The two main workflows are dialogue partitioning and human-evaluation sample selection.

Set these paths before running the Slurm wrappers:

```bash
export PROJECT_ROOT=/path/to/gesbench
export DATA_ROOT=/path/to/data/gestalt_bench
export APPTAINER_ROOT=/path/to/apptainers
```

Create the Slurm log directory before submitting from the repository root:

```bash
mkdir -p logs
```

## Dialogue Partitioning

Dialogue partitioning is implemented by `dialogue_partition.py` and wrapped by:

- `job_scripts/dialogue_partition_<cluster1>.sh`
- `job_scripts/dialogue_partition_<cluster2>.sh`
- `job_scripts/dialogue_partition_transfer.sh`

The input folder must contain videos named:

```text
dia<dialogue_id>_utt<utterance_id>.mp4
```

Example:

```bash
sbatch job_scripts/dialogue_partition_<cluster1>.sh --input-path mintrec2/raw --mode context --utt 1,2,3
```

The wrapper maps `--input-path dataset/subfolder` to:

```text
input:  ${DATA_ROOT}/dataset/subfolder
output: ${DATA_ROOT}/dataset/${mode}/subfolder
```

For each dialogue, the partitioner scans utterances in order and builds aligned three-utterance windows. With the default `--stride 2`, a processed window `[u_n, u_n+1, u_n+2]` is followed by the next window starting after two skipped utterances. From each complete window, it creates groups that share the same final utterance:

```text
1-utt: d<dialogue>u<end>
2-utt: d<dialogue>u<end-1>-u<end>
3-utt: d<dialogue>u<end-2>-u<end>
```

Output folders are:

```text
${DATA_ROOT}/${dataset}/${mode}/${subfolder}/1-utt_group/
${DATA_ROOT}/${dataset}/${mode}/${subfolder}/2-utt_group/
${DATA_ROOT}/${dataset}/${mode}/${subfolder}/3-utt_group/
```

`context` mode writes each group as a flat folder of cumulative clips. For multi-utterance groups, previous utterances are prepended to each generated clip so every clip contains the context plus the cumulative target utterance.

`nested` mode keeps previous utterances as whole videos in the group folder and writes cumulative clips for the final utterance in a nested subfolder.

Useful options:

```bash
--dialogue-range N     # process dialogue ids [(N-1)*100, N*100)
--clip-length SEC      # cumulative clip step, default 0.5
--mode context|nested
--cut SEC              # trim the target utterance before clipping
--utt 1,2,3            # choose which group sizes to materialize
--overwrite-1utt       # regenerate existing 1-utt outputs
```

Each run writes:

```text
partition_summary.json
partition_summary.txt
partition_groups.csv
partition_dialogues.csv
```

## Human-Evaluation Sample Selection

Human-evaluation sample packs are created by `select_human_eval_samples.py` and wrapped by:

```bash
job_scripts/select_human_eval_samples_<cluster1>.sh
```

This workflow samples from already partitioned `context` folders. The expected input layout is:

```text
${DATA_ROOT}/mintrec2/context/1-utt_group/
${DATA_ROOT}/mintrec2/context/2-utt_group/
${DATA_ROOT}/mintrec2/context/3-utt_group/
${DATA_ROOT}/meld/context/1-utt_group/
${DATA_ROOT}/meld/context/2-utt_group/
${DATA_ROOT}/meld/context/3-utt_group/
${DATA_ROOT}/seamless_interaction/context/1-utt_group/
${DATA_ROOT}/seamless_interaction/context/2-utt_group/
```

Run:

```bash
sbatch job_scripts/select_human_eval_samples_<cluster1>.sh
```

By default, it uses `--seed 42`, overwrites the output root, and writes to:

```text
${DATA_ROOT}/human_eval/samples
```

The selection policy is fixed in `DATASET_CONFIGS`:

```text
mintrec2:              160 samples, 4 batches, groups 1/2/3
meld:                  160 samples, 4 batches, groups 1/2/3
seamless_interaction:   80 samples, 2 batches, groups 1/2
```

A candidate is valid only when all required group-size folders exist for the same dialogue and ending utterance. Selected samples are copied into batch folders, and each batch folder is zipped.

The output layout is:

```text
${DATA_ROOT}/human_eval/samples/${dataset}/batch01/
${DATA_ROOT}/human_eval/samples/${dataset}/batch01.zip
${DATA_ROOT}/human_eval/samples/${dataset}/batch02/
${DATA_ROOT}/human_eval/samples/${dataset}/batch02.zip
```

The selector also writes:

```text
selection_summary.json
selection_summary.txt
```

To change the destination or seed:

```bash
sbatch job_scripts/select_human_eval_samples_<cluster1>.sh --output-root /path/to/output --seed 123
```

Use `--no-overwrite` to fail instead of replacing an existing output root.

## Other Utilities

- `dataset_info.py`: summarize prepared datasets and result folders.
- `seamless_construct_utterance.py`: construct utterance-level clips for the seamless interaction data.
- `seamless_construct_interaction.py`: construct grouped interaction clips.
- `overlap_check.py`: inspect overlap between prepared sample sets.
- `video_utils.py`: shared video cutting helpers used by the partitioner.

