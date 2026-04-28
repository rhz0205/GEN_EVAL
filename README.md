GEN_EVAL

## 项目定位

GEN_EVAL 是一个轻量级、离线、manifest 驱动的自动驾驶生成视频评估工具。

它当前聚焦于：

- 生成视频评估
- 本地路径驱动的离线运行
- 简单 CLI + 可复用源码包

它不是：

- 训练框架
- 重型实验管理平台
- 自动下载权重或自动部署环境的工具
- Ray / Rancher / Volcano / Kubernetes 集群管理器

## 项目结构

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

只保留 CLI 入口：

- `scripts/evaluate.py`：运行评估
- `scripts/inspect_manifest.py`：检查 manifest
- `scripts/manifest_from_pkl.py`：从 pkl 生成 manifest
- `scripts/summarize_results.py`：汇总结果

### `src/gen_eval/`

核心可复用源码包：

- `config.py`：读取 YAML 配置并解析 run config
- `dataset.py`：加载 manifest、提供基础检查
- `evaluator.py`：调度指标执行
- `execution.py`：本地 / 可选 Ray 执行后端
- `manifest_builder.py`：构建 manifest
- `registry.py`：指标注册表
- `result_summary.py`：结果摘要格式化
- `result_writer.py`：结果写盘
- `schemas.py`：样本结构

### `src/gen_eval/metrics/`

指标实现目录。当前 canonical metric names：

- `view_consistency`
- `temporal_consistency`
- `appearance_consistency`
- `depth_consistency`
- `semantic_consistency`
- `instance_consistency`

### `src/gen_eval/models/`

放评估指标依赖的轻量模型适配器代码，不放权重。

### `src/gen_eval/third_party/`

放 vendored 第三方源码，不放大模型权重或 checkpoint。

### `pretrained_models/`

放本地预训练权重与 checkpoint。不要把这些内容移动到 `src/` 下面。

## 配置说明

### 数据集配置

位于 `configs/datasets/*.yaml`。

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

位于 `configs/runs/*.yaml`。

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

可选字段：

- `run_name`

如果未提供，则默认使用 run config 文件名作为 `run_name`。

### 指标配置

统一放在 `configs/metrics.yaml`，尽量只暴露用户真正需要改的字段，例如：

- `repo_path`
- `weight_path`
- `model_path`
- `batch_size`
- `resize`

## 本地开发边界

本地机器可以没有：

- 真实数据集
- 视频文件
- 预训练权重
- CUDA
- Ray

因此，本地验证应限制为静态检查和 CLI 帮助信息：

```bash
python -m compileall scripts src
python scripts/evaluate.py --help
python scripts/inspect_manifest.py --help
python scripts/manifest_from_pkl.py --help
python scripts/summarize_results.py --help
```

如果本地缺少 `PyYAML`、`torch`、`LoFTR`、`DINO`、`DINOv2`、`Video-Depth-Anything`、本地权重等，应视为运行环境限制，而不是项目结构错误。

## 远程部署流程

真实评估应在物理隔离的远程集群上完成。推荐流程：

1. 获取最新代码

```bash
git pull origin main
```

2. 创建或激活 Python 环境

3. 安装远程环境所需依赖

4. 准备 manifest

5. 准备 `pretrained_models/`

6. 按远程路径修改配置：
   - `configs/datasets/*.yaml`
   - `configs/runs/*.yaml`
   - `configs/metrics.yaml`

7. 先检查 manifest

```bash
python scripts/inspect_manifest.py --manifest manifests/sample.json
```

8. 再运行评估

```bash
python scripts/evaluate.py --config configs/runs/sample.yaml
```

9. 检查输出目录和摘要文件

## Ray 使用说明

Ray 是可选后端，本地执行仍然是默认模式。

要点：

- Ray head 和 Ray workers 需要在 GEN_EVAL 之外启动
- GEN_EVAL 只会连接到已有 Ray 集群
- 不要把 Rancher / Volcano / Kubernetes 基础设施 YAML 放进这个仓库

Ray head 示例：

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0 --num-cpus=64 --num-gpus=0
```

Ray worker 示例：

```bash
RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
ray start --address=$RAY_HEAD:$RAY_HEAD_PORT --num-cpus=128 --num-gpus=8
```

run config 中按需加入：

```yaml
runtime:
  backend: ray
  ray_address: auto
  num_workers: 32
  device: cuda
```

## 远程调试建议

远程调试时，优先把以下信息带回本地做分析：

- 完整命令行
- 终端错误输出
- `summary.txt`
- `result.json`

以下问题应该明确暴露出来，而不是被静默忽略：

- manifest 缺失
- 视频路径缺失或不可读
- 权重缺失
- Python 依赖缺失
- Ray 连接失败

## 路径约定

- 不要在配置里写 Windows 专用绝对路径作为默认值
- 远程路径应通过 YAML 配置
- 预训练权重放在 `pretrained_models/`
- 第三方源码放在 `src/gen_eval/third_party/`
- 模型适配器代码放在 `src/gen_eval/models/`

## manifest 工作流

### 检查 manifest

```bash
python scripts/inspect_manifest.py --manifest manifests/sample.json
```

### 从 pkl 生成 manifest

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

## 输出目录约定

默认输出目录结构：

```text
outputs/{dataset_name}/{run_name}/{timestamp}/
```

其中：

- `dataset_name` 来自解析后的数据集配置
- `run_name` 优先读取 run config 中的 `run_name`，未提供时回退为配置文件名
- `timestamp` 使用 `YYYYMMDD_HHMMSS`

每次评估默认只写三个文件：

1. `result.json`
   - 机器可读完整结果
   - 包含 run、runtime、manifest、config 和 metrics
2. `summary.txt`
   - 远程终端易读摘要
3. `latest.txt`
   - 位于 `outputs/{dataset_name}/{run_name}/`
   - 内容是最近一次时间戳目录名

示例：

```text
outputs/sample_data/sample/20260428_153000/
  result.json
  summary.txt

outputs/sample_data/sample/latest.txt
```

这个结构的目的：

- 避免覆盖历史结果
- 方便远程多次对比
- 保持输出最小化
- 不引入重型实验管理框架

## 常用命令

### 运行评估

```bash
python scripts/evaluate.py --config configs/runs/sample.yaml
```

### 汇总结果

```bash
python scripts/summarize_results.py --result outputs/sample_data/sample/20260428_153000/result.json
```

## 添加新指标

建议步骤：

1. 在 `src/gen_eval/metrics/` 下新增指标文件
2. 保持接口为 `evaluate(samples) -> dict`
3. 在 `metrics/__init__.py` / `registry.py` 中注册
4. 在 `configs/metrics.yaml` 中加入配置
5. 在 `configs/runs/*.yaml` 的 `metrics:` 列表中启用

原则：

- 指标逻辑尽量放在指标模块内部
- 模型和权重懒加载
- 不要在 import 时加载大模型

## 当前重构状态

当前仓库已经完成这些方向上的整理：

- `scripts/` 保持为薄 CLI
- 复用逻辑收敛到 `src/gen_eval/`
- 执行后端独立在 `src/gen_eval/execution.py`
- 配置统一为 YAML
- manifest 和结果输出保持 JSON
- 输出目录采用轻量时间戳结构

## 注意事项

- 不要增加 `--dry-run` 或 `--check` 模式，除非明确要求
- 不要把这个项目扩展成 MLflow、W&B、数据库或 dashboard 系统
- 不要把权重提交到源码目录
- 不要把 `scripts/` 扩展成承载核心业务逻辑的地方
