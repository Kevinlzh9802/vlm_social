# Apptainer for Ming-Lite-Omni

This folder contains a reproducible Apptainer setup to run inference scripts in this repo.

## Files

- `Ming-lite-omni.def`: Apptainer definition file
- `requirements.txt`: Python dependencies for container runtime
- `build.sh`: Build helper script for `.sif`
- `run_infer.sh`: Runtime wrapper (`--nv`, bind repo to `/workspace`)

## Prerequisites

- NVIDIA driver on host
- `apptainer` installed on host
- GPU-enabled host (recommended)

## 1) Build image

Run from repo root:

```bash
bash apptainer/build.sh
```

If your host requires fakeroot for builds:

```bash
cd apptainer
apptainer build --fakeroot ming-lite-omni.sif Ming-lite-omni.def
```

## 2) Run inference scripts

Text/image/video/audio inference:

```bash
bash apptainer/run_infer.sh python test_infer.py
```

Image generation inference:

```bash
bash apptainer/run_infer.sh python test_infer_gen_image.py
```

Audio tasks (ASR + speech QA/TTS):

```bash
bash apptainer/run_infer.sh python test_audio_tasks.py
```

## Optional: install local Matcha wheel for audio generation

For `test_audio_tasks.py` TTS path, install the local wheel once inside the container:

```bash
bash apptainer/run_infer.sh pip install /workspace/data/matcha_tts-0.0.5.1-cp310-cp310-linux_x86_64.whl
```

## Notes

- Repo root is mounted into container at `/workspace`.
- The scripts default to model path `.`; run commands from repo root as shown above.
- For large models, ensure enough GPU memory (README indicates ~62 GB bf16 for full model load).
