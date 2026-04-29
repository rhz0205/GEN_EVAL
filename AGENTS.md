## Project Rules

1. `worldbench` is an offline evaluation framework for generated autonomous-driving videos.
2. It is not a training framework, not a heavy experiment-management platform, and not a dashboard system.
3. The canonical workflow is:
   data list
   -> dataset normalization
   -> reference data generation
   -> multi-dimensional evaluation modules
   -> result summarization
   -> visualization outputs
4. The only canonical config files are:
   `configs/dataset.yaml`
   `configs/metrics.yaml`
   `configs/reference.yaml`
   `configs/run.yaml`
5. Canonical dataset names are:
   `cosmos`
   `geely`
   `real`
   `sample`
6. Data file naming rule:
   `data/{dataset_name}_{data_count}_{timestamp}.json`
7. Output directory rule:
   `outputs/{dataset_name}/{data_count}_{timestamp}/`
8. Output layout:
   `results/`
   `logs/`
   `visualizations/`
9. Visualization folders under `visualizations/` must be exactly:
   `depth_raw`
   `semantic_raw`
   `multiview_match_raw`
   `depth_6v_image`
   `semantic_6v_image`
   `multiview_match_6v_image`
   `depth_6v_video`
   `semantic_6v_video`
   `multiview_match_6v_video`
10. Do not create `src/gen_eval/`.
11. Do not create `src/gen_eval/metrics/`.
12. All metric implementations belong in `src/modules/`.
13. Modules must contain real implementations, not wrappers around any metrics package.
14. Do not introduce `XXXMetric` legacy class names in the final structure.
15. Use concise class names:
    `VideoIntegrity`
    `TemporalConsistency`
    `AppearanceConsistency`
    `DepthConsistency`
    `SemanticConsistency`
    `InstanceConsistency`
    `ViewConsistency`
16. The dataset layer belongs under `src/dataset/`.
17. Do not create `src/gen_eval/dataset.py`.
18. Do not create `src/gen_eval/reference.py`.
19. The reference layer belongs under `src/reference/`.
20. The orchestration layer belongs under `src/models/`.
21. The visualization layer belongs under `src/visualization/`.
22. Shared schemas belong in `src/schemas.py`.
23. Do not create old `src/gen_eval/evaluator.py`, `src/gen_eval/execution.py`, or `src/gen_eval/registry.py`.
24. Scripts should be thin CLI entrypoints.
25. Scripts should add the project `src/` directory to `sys.path` when needed.
26. Scripts should import from the flat packages, for example:
    `from schemas import GenerationSample`
    `from dataset import build_dataset`
    `from modules import build_module`
    `from reference import ReferencePreparer`
    `from models import GenEval`
27. Do not use imports like:
    `from gen_eval.dataset import ...`
    `from gen_eval.modules import ...`
    `from gen_eval.reference import ...`
    `from gen_eval.models import ...`
28. Core logic should live under `src/`.
29. The reference layer should generate reference files before metric evaluation.
30. OpenSeeD should be part of the reference layer, not `semantic_consistency.py`.
31. `semantic_consistency.py` should only read prepared semantic masks.
32. OpenSeeD semantic reference protocol:
    `metadata.semantic_masks`
    `metadata.semantic_num_classes`
    `metadata.semantic_ignore_label`
33. Use `ignore_label = -1` for ignored semantic pixels.
34. For semantic consistency, fixed weights are:
    `S_SemC = 0.5 * S_LFR + 0.4 * S_SAC + 0.1 * S_CDS`
35. For no-reference temporal or appearance consistency, preferred aggregation is:
    `score = ACM / (1 + TJI)`
36. Do not change metric formulas unless explicitly requested.
37. Config field names should be unified:
    `weight_path`
    `repo_path`
    `model_path`
    `config_path`
    `device`
    `batch_size`
38. Avoid legacy config aliases unless explicitly needed.
39. Comments should be minimal.
40. If comments are necessary, use Chinese block-level comments.
41. Do not add unnecessary English implementation docstrings.
42. Do not add redundant comments such as:
    `# type: ignore`
    `# noqa: BLE001`
    `# type: ignore[assignment]`
43. Future Codex tasks should:
    - inspect old source only as reference
    - write only into `D:\Project\worldbench`
    - make limited changes
    - report touched files, risks, and validation commands
44. Remote-server collaboration control:
    - all future code or config changes must be confirmed by the user before modification
    - after each approved change, always report the touched files so the user can copy them to the remote server
    - prefer relative paths for all non-data resources
    - when reasoning about remote deployment, use `/di/group/renhongze/wm_gen_data_eval` as the project root
45. Project execution workflow for the current phase:
    - random sample selection
    - dataset inspect
    - reference data preparation
    - multi-dimensional evaluation
    - result summarization
    - visualization
46. The current development roadmap must follow these phases:
    - phase 1: single-machine end-to-end pipeline validation
    - phase 2: metric logic review and correction
    - phase 3: multi-server multi-gpu parallelization based on ray
    - phase 4: overall project optimization
47. Phase 1 requirements:
    - first prioritize a stable single-machine closed loop before changing metric logic
    - the closed loop includes random sampling, inspect, reference preparation, evaluation, summarization, and visualization
    - use small sample sets first, then gradually increase data scale
    - prefer fixing protocol, path, dependency, and integration issues before optimizing algorithms
48. Phase 1 completion criteria:
    - the full pipeline can run on the remote server from sampling to visualization
    - reference outputs are generated as formal artifacts, not temporary byproducts
    - result files are written stably under the canonical output layout
    - failed or skipped samples are traceable from structured result files
49. Phase 1 diagnostic priority:
    - always inspect `failed_samples.json` first when module execution fails
    - always distinguish initialization failures, per-sample failures, and skipped-sample cases
    - do not start ray-based parallelization before the single-machine pipeline is stable
50. Phase 2 requirements:
    - review whether each metric definition matches the intended evaluation target
    - review whether each score range, aggregation rule, and failure policy is reasonable
    - change metric formulas only when explicitly approved by the user
51. Phase 3 requirements:
    - parallelization should be introduced only after the single-machine logic is stable
    - ray-based scaling should consider sample-level parallelism, module-level parallelism, and staged reference/evaluation execution
    - do not parallelize unstable or poorly understood logic
52. Phase 4 requirements:
    - optimize config clarity, dependency handling, logging, output protocol stability, and performance
    - preserve the canonical workflow and directory conventions while optimizing
53. The current formal metric-role definitions for the validated baseline are:
    - `video_integrity`: input health and multi-view package integrity check
    - `temporal_consistency`: RGB-domain temporal stability metric
    - `depth_consistency`: depth-based temporal stability metric
54. `video_integrity` should be interpreted as a gatekeeping health-check metric:
    - it evaluates whether the expected multi-view videos are present, readable, and mutually consistent in frame count, fps, duration, and resolution
    - it is not a perceptual quality metric
    - a successful module run does not imply every sample passed the integrity checks
55. `temporal_consistency` should be interpreted as the primary no-reference temporal stability metric:
    - it evaluates temporal smoothness in the RGB or perceptual feature domain
    - it is intended to reflect visual flicker, abrupt temporal changes, and unstable appearance over time
56. `depth_consistency` should be interpreted as depth-based temporal stability, not generic geometric correctness:
    - it evaluates temporal stability after projecting the video into a depth-related representation
    - it is intended to complement `temporal_consistency`, not replace it
    - it should not be described as a direct measure of absolute depth accuracy or full cross-view geometric consistency unless the implementation is explicitly changed later
57. The intended division of labor among these three metrics is:
    - `video_integrity`: verifies whether the sample package is structurally usable for downstream evaluation
    - `temporal_consistency`: measures temporal stability in the visual feature domain
    - `depth_consistency`: measures temporal stability in the depth-related feature domain
58. During future reviews and documentation updates:
    - keep the metric name `depth_consistency` for now
    - explicitly explain in prose that its current meaning is `depth-based temporal stability`
    - do not claim that `temporal_consistency` and `depth_consistency` are duplicates; their distinction is the feature domain they operate in
