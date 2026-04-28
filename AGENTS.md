GEN_EVAL project rules:

GEN_EVAL is a lightweight, manifest-driven, offline evaluation toolkit for multi-view autonomous-driving generated videos. It is adapted from WorldLens generation metrics, but it must not revert to WorldLens-style engineering.

Hard constraints:
- Do not revert to WorldLens-style `method_name/generated_results/video_submission/__call__` APIs.
- Do not use `worldbench.utils.common` or `video_relative` in `src/gen_eval`.
- Do not use `git clone`, `wget`, or online downloads.
- Do not install packages.
- Do not modify `pretrained_models` content.
- Do not run 1k or full-scale evaluations unless explicitly requested.
- Metrics must expose `evaluate(samples) -> dict`.
- Inputs must be manifest-driven.
- Model paths must be local and configurable.
- If a dependency, model, or path is missing, return skipped/failed instead of crashing the whole evaluation.

Current canonical dataset groups:
- `sample_data`
- `geely_data`
- `cosmos_data`
- `real_data`

Do not use `debug` as a dataset group name.

Current canonical metric names:
- `view_consistency`
- `temporal_consistency`
- `appearance_consistency`
- `depth_consistency`
- `semantic_consistency`
- `instance_consistency`

Current preferred project layout:

```text
configs/
  datasets/
  metrics.yaml
  runs/

manifests/
  sample.json
  geely.json
  cosmos.json
  real.json

outputs/
  sample_data/
  geely_data/
  cosmos_data/
  real_data/

scripts/
  evaluate.py
  manifest_from_pkl.py
  inspect_manifest.py
  summarize_results.py

src/gen_eval/
  __init__.py
  schemas.py
  dataset.py
  evaluator.py
  registry.py
  result_writer.py
  metrics/

src/third_party/
  video_depth_anything/
```

Minimal core package responsibilities:
- `schemas.py`: sample schema
- `dataset.py`: manifest loading
- `evaluator.py`: runs enabled metrics
- `registry.py`: maps canonical metric names to metric classes
- `result_writer.py`: writes result JSON
- `metrics/`: metric implementations

Keep metric-specific loading logic inside each metric file:
- LoFTR logic inside `view_consistency.py`
- CLIP logic inside `temporal_consistency.py`
- DINO logic inside `appearance_consistency.py`
- Video-Depth-Anything / DINOv2 logic inside `depth_consistency.py`

Do not create or expand extra package layers unless explicitly requested:
- Do not create `src/gen_eval/models/`.
- Do not expand `src/gen_eval/utils/` unless absolutely required by existing imports.

Config rules:

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

Naming rules:
- manifest filenames under `manifests/` must not add `_manifest`
- run names must not use `_all`
- output naming is `outputs/{dataset_group}/{run_name}`

Examples:
- `manifests/sample.json`
- `configs/runs/sample.yaml`
- `configs/runs/sample_view.yaml`
- `outputs/sample_data/sample`
- `outputs/geely_data/geely`

Dependency-checking scope rules:

Codex should only inspect dependencies between existing project files and packages inside this repository.

Focus on:
- whether `src/gen_eval` modules import each other correctly
- whether canonical metric modules are referenced correctly
- whether registry mappings are consistent
- whether config metric keys match registered metric names
- whether scripts import project modules correctly
- whether old metric names only remain as backward-compatible aliases

Codex must not treat missing local runtime dependencies as project errors. In this local workspace, the full evaluation environment is not configured.

Do not report these as code problems unless they are directly caused by project imports:
- missing `torch`
- missing `torchvision`
- missing `numpy`
- missing `scipy`
- missing `skimage`
- missing `cv2`
- missing `PIL`
- missing `clip`
- missing `open_clip`
- missing `transformers`
- missing `LoFTR`
- missing `DINO`
- missing `DINOv2`
- missing `Video-Depth-Anything`
- missing `xFormers`
- missing `CUDA`
- missing `PyYAML`
- missing local model weights
- invalid local absolute weight paths
- unavailable `pretrained_models` content

Do not:
- install packages
- download weights
- run environment-dependent metric execution
- run long evaluations
- modify `pretrained_models`
- modify weight paths just because they are placeholders

Allowed checks:
- static import/path checks within the project where optional external dependencies are not required
- YAML config syntax validation
- registry key validation
- file existence checks for project files only
- `grep` or `rg` checks for old names, deprecated imports, and WorldLens-style APIs

If an import check fails because of an optional external package, classify it as:
- `environment/runtime dependency not available in local workspace`

When reporting validation results, separate:
1. project dependency issues
2. config/schema issues
3. environment/runtime dependency limitations

Do not propose environment setup unless explicitly asked.
