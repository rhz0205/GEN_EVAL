GEN_EVAL

## 项目定位

GEN_EVAL 是一个面向自动驾驶多视角生成视频的轻量级、离线、manifest 驱动评估工具。

它当前专注于：
- 评估生成结果
- 本地路径与本地配置驱动
- 轻量脚本入口 + 可复用源码包

它**不是**：
- 训练框架
- 重型实验管理平台
- 自动下载模型或自动安装环境的工具
- WorldLens 运行时环境的复刻版本

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
      manifest_builder.py
      registry.py
      result_summary.py
      result_writer.py
      schemas.py
      metrics/
      models/
      third_party/
```

## 各目录职责

### `scripts/`

只保留 CLI 入口，不放可复用业务逻辑。

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
- `manifest_builder.py`：从 pkl 构建 manifest 与统计信息
- `registry.py`：指标注册表
- `result_summary.py`：结果摘要与命令行输出格式化
- `result_writer.py`：结果 JSON 输出与路径辅助
- `schemas.py`：样本与对象轨迹结构

### `src/gen_eval/metrics/`

放具体评估指标实现。当前指标命名采用统一的 canonical names：
- `view_consistency`
- `temporal_consistency`
- `appearance_consistency`
- `depth_consistency`
- `semantic_consistency`
- `instance_consistency`

### `src/gen_eval/models/`

放评估指标依赖的**轻量模型适配器**。这里应该是小型桥接代码，而不是大而全的模型框架。

不要把权重放在这里。

### `src/gen_eval/third_party/`

放 vendored 第三方源码，仅用于保留项目内需要引用的外部代码副本。

不要把 checkpoint、`.pth`、`.ckpt`、`.bin`、`.onnx`、`.safetensors` 这类模型文件放在这里。

### `pretrained_models/`

放本地模型权重与第三方预训练文件。

这是权重目录，不是源码目录。

### `configs/`

使用 YAML：
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
- 配置已统一为 YAML
- manifest 与结果输出仍保持 JSON
- WorldLens 运行时依赖已不再作为当前主路径的一部分

当前仍需注意：
- 本仓库默认是离线使用方式
- 没有自动安装依赖
- 没有自动下载权重
- 缺失 `PyYAML`、`torch`、`LoFTR`、`DINO`、`DINOv2`、`Video-Depth-Anything`、本地权重等，应该视为运行环境限制，而不是项目结构错误

## 注意事项

- 不要把模型权重提交到源码目录
- 不要把 `scripts/` 再扩展成承载核心业务逻辑的地方
- 不要把 GEN_EVAL 重新做成重型平台
- 如果只是增加一个新指标，优先沿用现有 registry + config + script 结构
