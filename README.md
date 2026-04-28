GEN_EVAL

GEN_EVAL is a lightweight, offline evaluation toolkit for multi-view autonomous-driving generated videos. It is adapted from WorldLens generation-evaluation ideas, but uses a manifest-driven workflow and a minimal project structure.

**Scope**
- generation evaluation only
- offline, local-path-based workflow
- manifest-driven sample loading
- lightweight metric registry and evaluator

**Non-goals**
- reconstruction evaluation
- downstream task evaluation
- WorldLens runtime reproduction
- automatic package installation or model download

**Minimal Structure**
- `configs/datasets/`: dataset configs
- `configs/metrics.yaml`: merged metric configs
- `configs/runs/`: run configs
- `manifests/`: dataset manifests
- `outputs/`: saved evaluation outputs
- `scripts/`: CLI entrypoints
- `src/gen_eval/`: core package
- `src/third_party/video_depth_anything/`: external depth code location

Core package files:
- `src/gen_eval/schemas.py`: defines the sample schema
- `src/gen_eval/dataset.py`: loads manifest JSON into samples
- `src/gen_eval/evaluator.py`: runs enabled metrics
- `src/gen_eval/registry.py`: maps canonical metric names to metric classes
- `src/gen_eval/result_writer.py`: writes result JSON
- `src/gen_eval/metrics/`: metric implementations

**Canonical Dataset Groups**
- `sample_data`
- `geely_data`
- `cosmos_data`
- `real_data`

Do not use `debug` as a dataset group name.

**Canonical Metric Names**
- `view_consistency`
- `temporal_consistency`
- `appearance_consistency`
- `depth_consistency`
- `semantic_consistency`
- `instance_consistency`

**Config Design**

Dataset config schema:

```yaml
dataset_name: sample_data
manifest_path: manifests/sample.json
description: Small-batch sample dataset for fast metric validation.
default_output_dir: outputs/sample_data
```

Metric config rules:
- use one merged `configs/metrics.yaml`
- top-level keys are canonical metric names
- keep only practical user-facing fields
- do not require:
  - `mode`
  - `backend`
  - `device`
  - `score_key`
  - `use_all_views`

Run config schema:

```yaml
dataset: sample_data
metrics:
  - view_consistency
  - temporal_consistency
  - appearance_consistency
  - depth_consistency
runtime:
  device: cuda
```

Run config inference rules:
- `run_name` comes from the run config filename stem
- `dataset_config_path` resolves to `configs/datasets/{dataset}.yaml`
- `metric_config_path` is always `configs/metrics.yaml`
- `output_dir` resolves to `outputs/{dataset}/{run_name}`
- `save_details` defaults to `true`

**Naming Rules**

Manifest naming examples:
- `manifests/sample.json`
- `manifests/geely.json`
- `manifests/cosmos.json`
- `manifests/real.json`

Run naming examples:
- `configs/runs/sample.yaml`
- `configs/runs/sample_view.yaml`
- `configs/runs/sample_temporal.yaml`
- `configs/runs/sample_appearance.yaml`
- `configs/runs/sample_depth.yaml`
- `configs/runs/geely.yaml`
- `configs/runs/cosmos.yaml`
- `configs/runs/real.yaml`

Output naming rule:
- `outputs/{dataset_group}/{run_name}`

Examples:
- `outputs/sample_data/sample`
- `outputs/sample_data/sample_view`
- `outputs/geely_data/geely`

**Main Commands**

Inspect the sample manifest:

```bash
python scripts/inspect_manifest.py --manifest manifests/sample.json
```

Run sample view evaluation:

```bash
python scripts/evaluate.py --config configs/runs/sample_view.yaml
```

Run sample comprehensive evaluation:

```bash
python scripts/evaluate.py --config configs/runs/sample.yaml
```

Summarize a saved result:

```bash
python scripts/summarize_results.py --result outputs/sample_data/sample/result.json
```

Replace the result path with the actual saved result JSON under `outputs/sample_data/sample/` if the filename is timestamped or different.

Generate a default sample manifest:

```bash
python scripts/manifest_from_pkl.py \
  --pkl /path/to/data.pkl \
  --dataset-name sample_data \
  --dataset-split sample \
  --sample-total 20 \
  --seed 42 \
  --detect-camera-videos \
  --primary-camera camera_front
```

This writes:
- `manifests/sample.json`
- `outputs/sample_data/sample_manifest_stats.json`

**Dependency Scope**
- GEN_EVAL is an offline project.
- Do not auto-install packages.
- Do not auto-download weights.
- PyYAML is required for reading YAML configs.
- Missing `torch`, `clip`, `DINO`, `LoFTR`, `DINOv2`, `Video-Depth-Anything`, `CUDA`, or local weights should be treated as runtime environment limitations, not project-structure bugs.

**Known Limitations**
- `fvd` is not part of the current no-reference main workflow.
- `semantic_consistency` is minimal unless prepared semantic metadata is available.
- `appearance_consistency` measures DINO-based full-frame visual appearance stability and does not require object tracks.
- `instance_consistency` is minimal unless prepared object metadata is available.
- `instance_consistency` measures instance / track-level stability and requires object metadata.
