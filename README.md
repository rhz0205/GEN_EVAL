GEN_EVAL

## 项目定位

GEN_EVAL 是一个轻量级、离线、manifest 驱动的自动驾驶生成视频评估工具。

当前项目专注于：
- 生成结果评估
- 本地路径与本地配置驱动
- 轻量脚本入口 + 可复用源码包

当前项目**不是**：
- 训练框架
- 重型实验管理平台
- 自动安装环境或自动下载模型的工具
- WorldLens 运行时环境的复刻版

## 推荐目录结构

```text
GEN_EVAL/
  configs/
    datasets/
    runs/
    metrics.yaml
  manifests/
  outputs/
  pretrained_models/
  scripts/
    evaluate.py
    inspect_manifest.py
    manifest_from_pkl.py
    summarize_results.py
  src/
    gen_eval/
      __init__.py
      config.py
      dataset.py
      evaluator.py
      execution.py
      manifest_builder.py
      registry.py
      result_summary.py
      result_writer.py
      schemas.py
      metrics/
      models/
      third_party/
```

## 目录职责

### `scripts/`

只保留 CLI 入口，不承载可复用业务逻辑。

当前脚本：
- `scripts/evaluate.py`：运行评估
- `scripts/inspect_manifest.py`：检查 manifest
- `scripts/manifest_from_pkl.py`：从 pkl 生成 manifest
- `scripts/summarize_results.py`：汇总结果 JSON

### `src/gen_eval/`

核心源码包，承载可复用逻辑。

主要模块：
- `config.py`：读取 YAML 配置，解析 run config
- `dataset.py`：加载 manifest、提取样本、输出基础 manifest 摘要
- `evaluator.py`：按配置调度指标执行
- `execution.py`：执行后端选择，默认本地执行，可选 Ray
- `manifest_builder.py`：从 pkl 构建 manifest 与统计信息
- `registry.py`：指标注册表
- `result_summary.py`：结果摘要与命令行输出格式化
- `result_writer.py`：结果 JSON 输出与路径辅助
- `schemas.py`：样本与对象轨迹结构

### `src/gen_eval/metrics/`

放具体评估指标实现。当前 canonical metric names：
- `view_consistency`
- `temporal_consistency`
- `appearance_consistency`
- `depth_consistency`
- `semantic_consistency`
- `instance_consistency`

指标返回结果应保持稳定的字典结构，至少包含：
- `metric`
- `score`
- `num_samples`
- `details`
- `status`
- `reason`（当跳过或失败时）

这样本地执行与可选 Ray 执行都可以用同一套方式聚合结果。

### `src/gen_eval/models/`

放评估指标依赖的**轻量模型适配器**。这里应当是小型桥接代码，而不是完整模型框架。

不要把模型权重放在这里。

### `src/gen_eval/third_party/`

放 vendored 第三方源码，仅用于保留项目内需要引用的外部代码副本。

不要把 checkpoint、`.pth`、`.ckpt`、`.bin`、`.onnx`、`.safetensors` 这类模型文件放在这里。

### `pretrained_models/`

放本地模型权重与第三方预训练文件。

这是权重目录，不是源码目录。

### `configs/`

统一使用 YAML：
- `configs/datasets/`：数据集级别元信息
- `configs/runs/`：最小化 run 选择器
- `configs/metrics.yaml`：统一指标参数配置

### `manifests/`

放样本清单，保持 JSON。

manifest 是评估输入，不是配置文件。

### `outputs/`

放评估结果 JSON、manifest 统计文件等输出内容。

## 配置约定

### 数据集配置

示例：

```yaml
dataset_name: sample_data
manifest_path: manifests/sample.json
description: Small-batch sample dataset for fast metric validation.
default_output_dir: outputs/sample_data
```

当前数据集分组：
- `sample_data`
- `geely_data`
- `cosmos_data`
- `real_data`

### 运行配置

示例：

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

推导规则：
- `run_name` 由文件名推导
- dataset config 由 `configs/datasets/{dataset}.yaml` 推导
- metric config 固定为 `configs/metrics.yaml`
- `output_dir` 推导为 `outputs/{dataset}/{run_name}`

### 指标配置

统一放在 `configs/metrics.yaml` 中。对用户暴露的字段尽量保持通用，例如：
- `repo_path`
- `weight_path`
- `model_path`
- `batch_size`
- `resize`

避免把过多实现细节直接暴露在用户配置里。

## 执行模式

### 本地执行

默认使用本地执行，不需要 Ray。

示例：

```yaml
dataset: sample_data
metrics:
  - view_consistency
runtime:
  device: cuda
```

说明：
- 如果没有显式设置 `runtime.backend`，默认就是 `local`
- 如果没有显式设置 `runtime.num_workers`，默认使用安全的本地值，不需要用户额外配置

### 可选 Ray 执行

如果需要做样本级并行评估，可以在现有 run config 的 `runtime` 中显式加入：

```yaml
dataset: sample_data
metrics:
  - view_consistency
  - temporal_consistency
runtime:
  backend: ray
  ray_address: auto
  num_workers: 32
  device: cuda
```

说明：
- `backend` 默认是 `local`
- 只有当显式设置为 `ray` 时，才会尝试导入 Ray
- `ray_address` 仅在 `backend: ray` 时才需要；未设置时默认使用 `auto`
- 不需要为 Ray 单独新建 `sample_ray.yaml` 这类配置文件
- 直接把上面的 `runtime` 片段复制到现有 run config 里即可

集群侧的最小使用方式：
- 在开发节点启动 Ray head
- 在计算节点加入 Ray workers
- 用 `runtime.backend=ray` 运行 GEN_EVAL

当前 Ray 模式保持轻量：
- 不自动启动 head / worker
- 不在 Python 中提交 Rancher、Kubernetes、Volcano 作业
- GEN_EVAL 只通过 `ray.init(address=...)` 连接到已存在的 Ray 集群
- 不要求改写现有 metric 实现

## 基本使用流程

1. 准备或生成 manifest
2. 选择 run config
3. 运行评估
4. 汇总结果

## 常用命令

### 1. 检查 manifest

```bash
python scripts/inspect_manifest.py --manifest manifests/sample.json
```

### 2. 从 pkl 生成 manifest

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

默认会生成：
- `manifests/sample.json`
- `outputs/sample_data/sample_manifest_stats.json`

### 3. 运行评估

```bash
python scripts/evaluate.py --config configs/runs/sample.yaml
```

### 4. 汇总结果

```bash
python scripts/summarize_results.py --result outputs/sample_data/sample/<result>.json
```

请把 `<result>.json` 替换为实际保存的结果文件名。

## 如何添加新指标

建议步骤：
1. 在 `src/gen_eval/metrics/` 下新增一个指标模块
2. 保持接口为 `evaluate(samples) -> dict`
3. 在 `registry.py` / `metrics/__init__.py` 中注册
4. 在 `configs/metrics.yaml` 中加入该指标的配置项
5. 在某个 run config 的 `metrics:` 列表中启用它

原则：
- 指标逻辑尽量留在指标文件内部
- 懒加载模型和权重
- 不要在 import 时加载大模型

## 如何添加新的 manifest

有两种常见方式：

### 方式一：直接准备 JSON manifest

保持 manifest schema 与 `GenerationSample` 一致，核心字段包括：
- `sample_id`
- `generated_video`
- `reference_video`
- `prompt`
- `objects`
- `metadata`

### 方式二：从 pkl 生成

使用 `scripts/manifest_from_pkl.py`，按需指定：
- `--sample-total`
- `--sample-per-key`
- `--dataset-name`
- `--dataset-split`
- `--output`

## 当前重构状态

当前代码已经完成这些方向上的整理：
- `scripts/` 主要保留为薄 CLI 入口
- 可复用逻辑逐步收敛到 `src/gen_eval/`
- 执行后端已拆分为独立的 `src/gen_eval/execution.py`
- 配置已统一为 YAML
- manifest 与结果输出仍保持 JSON
- WorldLens 运行时依赖已不再作为当前主路径的一部分

当前仍需注意：
- 本仓库默认是离线使用方式
- 没有自动安装依赖
- 没有自动下载权重
- 缺失 `PyYAML`、`torch`、`LoFTR`、`DINO`、`DINOv2`、`Video-Depth-Anything`、本地权重等，应视为运行环境限制，而不是项目结构错误

## 注意事项

- 不要把模型权重提交到源码目录
- 不要把 `scripts/` 再扩展成承载核心业务逻辑的地方
- 不要把 GEN_EVAL 重新做成重型平台
- 如果只是增加一个新指标，优先沿用现有 registry + config + script 结构
