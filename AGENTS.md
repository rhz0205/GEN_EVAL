GEN_EVAL project rules:

GEN_EVAL is a lightweight, manifest-driven, offline evaluation toolkit for multi-view autonomous-driving generated videos. It should stay practical and flat, and must not drift back into WorldLens-style engineering.

Hard constraints:
- Do not reintroduce WorldLens-style `method_name/generated_results/video_submission/__call__` APIs.
- Do not use `worldbench.utils.common` or `video_relative` in `src/gen_eval`.
- Do not use online downloads such as `git clone`, `wget`, or model fetch scripts.
- Do not install packages.
- Do not modify `pretrained_models` content.
- Do not run full-scale evaluations unless explicitly requested.
- Metrics must expose `evaluate(samples) -> dict`.
- Inputs must be manifest-driven.
- Model paths must be local and configurable.
- If a dependency, model, or local path is missing, return skipped/failed instead of crashing the whole evaluation.

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

outputs/
  sample_data/
  geely_data/
  cosmos_data/
  real_data/

pretrained_models/

scripts/
  evaluate.py
  manifest_from_pkl.py
  inspect_manifest.py
  summarize_results.py

src/gen_eval/
  __init__.py
  config.py
  dataset.py
  evaluator.py
  manifest_builder.py
  registry.py
  result_summary.py
  result_writer.py
  schemas.py
  metrics/
  models/
  third_party/
```

Package responsibilities:
- `scripts/`: thin CLI entrypoints only
- `src/gen_eval/config.py`: YAML loading and run-config resolution
- `src/gen_eval/dataset.py`: manifest loading and lightweight manifest inspection helpers
- `src/gen_eval/evaluator.py`: runs enabled metrics
- `src/gen_eval/manifest_builder.py`: reusable manifest generation helpers
- `src/gen_eval/registry.py`: maps canonical metric names to metric classes
- `src/gen_eval/result_summary.py`: reusable result summary formatting
- `src/gen_eval/result_writer.py`: result JSON writing and output path helpers
- `src/gen_eval/schemas.py`: sample schema
- `src/gen_eval/metrics/`: metric implementations
- `src/gen_eval/models/`: lightweight model adapters used by metrics
- `src/gen_eval/third_party/`: vendored third-party source code only
- `pretrained_models/`: local pretrained weights and checkpoints

Boundary rules:
- Do not place model weights under `src/gen_eval/models/`.
- Do not place checkpoints under `src/gen_eval/third_party/`.
- Keep files like `.pth`, `.ckpt`, `.bin`, `.onnx`, and `.safetensors` under `pretrained_models/` or another explicit non-source local path.
- Keep metric-specific loading logic inside each metric module unless there is a clear need for a tiny reusable adapter.

Keep metric-specific logic local when practical:
- LoFTR logic inside `view_consistency.py`
- CLIP logic inside `temporal_consistency.py`
- DINO logic inside `appearance_consistency.py`
- Video-Depth-Anything / DINOv2 logic inside `depth_consistency.py`

Keep the project lightweight:
- Do not introduce deep package layering unless explicitly requested.
- Do not create a `backbone/` directory.
- Do not expand `src/gen_eval/utils/` into a large helper tree.
- Prefer small reusable functions over framework-like abstractions.

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

Dependency-checking scope rules:

Codex should focus on repository-internal correctness:
- whether `src/gen_eval` modules import each other correctly
- whether canonical metric modules are referenced correctly
- whether registry mappings are consistent
- whether config metric keys match registered metric names
- whether scripts import project modules correctly

Codex must not treat missing local runtime dependencies as project errors unless they are caused by project imports.

Do not report these as code problems by default:
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

Allowed checks:
- static import/path checks within the project
- YAML config syntax validation
- registry key validation
- file existence checks for project files only
- grep/rg checks for old names, deprecated imports, and WorldLens-style APIs

If an import or execution check fails because of an optional external package, classify it as:
- `environment/runtime dependency not available in local workspace`

When reporting validation results, separate:
1. project dependency issues
2. config/schema issues
3. environment/runtime dependency limitations

Do not propose environment setup unless explicitly asked.
