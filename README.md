# replay

## 项目概述

`replay` 是图像模态的 KDC 复现工程，覆盖语义桶、区域重要性、任务感知 JPEG 编码、随机丢包、知识库条件恢复和下游评测。实验分为 Cityscapes 语义分割与 ImageNet-1K 图像分类。代码以论文明确写出的组件为依据；论文未公开的量化、分包、随机信道等细节由项目配置和实现补足。

`results/figures/` 保存四张当前汇报图；`results/reports/` 保存重绘所需的统计数据。图 7(a)、7(b) 和 8(b) 是 **300 张 ImageNet 验证图的试跑**；图 8(a) 是 **100 张 Cityscapes val 图的试跑**，采用 30 kB、随机 GE 丢包、无 RST，并使用真值标签确定重点区域。它不是完整验证集结果，GT 重点区域也不是实际发送端的 CLIPSeg 预测。图 8(a) 展示逐图 IoU 分布，与聚合 IoU 的数值口径不同。

| 图 | 指标与对比 |
| --- | --- |
| 7(a) | 不同丢包率下的分类 Top-1：ProgJPEG、LimitNet、KDC |
| 7(b) | 不同传输预算下的分类 Top-1：JPEG、ProgJPEG、LimitNet、KDC |
| 8(a) | Car、Pedestrians、Vegetation 的逐图 IoU：LimitNet、KDC |
| 8(b) | 不同分辨率下的分类 Top-1 和 PSNR：LimitNet、KDC |

## 逻辑架构

```text
configs/ 配置
  ├─ src/replay/data/             读取 COCO/FSS、Cityscapes、ImageNet；生成清单与令牌缓存
  ├─ src/replay/knowledge/        SharedKB / PosteriorKB；语义桶与文本、图像向量
  ├─ src/replay/transmission/     CLIPSeg 候选区域 → 区域重要性 → 自适应量化/JPEG → 分包与随机丢包
  ├─ src/replay/receiver/         FiLM-UNet 接收端训练和恢复
  ├─ src/replay/baselines/        JPEG / ProgJPEG / LimitNet 对照
  └─ src/replay/evaluation/       分类、分割、码率与图表指标
scripts/                          依次调用上述模块，准备资源、训练、评测和汇总
results/reports/                  本次四张图的统计输入
results/figures/                  本次四张展示图
```

主要调用顺序：

1. `scripts/prepare_pilot_fold0.py`、`scripts/prepare_pilot_bf16.py` 准备 COCO-20 第 0 折 60 个训练类与 FSS-1000 非测试 760 类；`scripts/build_receiver_kb.py` 和 `scripts/cache_receiver_tokens.py` 准备接收端先验与 token 缓存；`scripts/train_pilot_15ep_ddp.py` 经 `replay.receiver` 训练恢复网络。已有旧版预训练 KB 时也可用 `scripts/prepare_pilot_kb.py` 增量补充 FSS 240 个非测试类。
2. `scripts/build_cityscapes_kb.py` 读取 Cityscapes train 参考图像和类别定义，调用 `replay.knowledge`；`replay.knowledge.bins` 生成并归一化语义桶。常规发送端在 `replay.transmission.importance` 中比较 CLIPSeg 候选区域特征与语义桶文本嵌入，再由 `replay.transmission.jpeg` 量化、编码。图 8(a) 当前试跑改用 val 真值重点区域。
3. `scripts/run_figure7_pilot_queue.py` 调用 `scripts/run_figure7_pilot_point.py`，经 `replay.evaluation.figure7_*` 评测图 7；`scripts/run_figure8a_gt_random_val.py` 经 `replay.evaluation.figure8a_gt_random` 评测图 8(a)；`scripts/run_figure8b_pilot_sweep.py` 与两个 1024 像素入口评测图 8(b)。
4. `scripts/replot_results.py` 仅读取 `results/reports/`，调用三处绘图函数生成四张 PNG，不加载模型或数据集。

配置位于 `configs/`；不同实验的预算、分包和模型路径以实际运行参数与报告 JSON 为准。`LICENSES/LimitNet-MIT.txt` 保存所用 LimitNet 代码的许可证文本。

## 快速部署和运行

以下命令在 Linux 服务器的项目根目录执行。Python 3.10+、CUDA 和 PyTorch GPU 环境由服务器准备；根据显卡驱动安装对应的 PyTorch 构建。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test,imagenet]"
export PYTHONPATH="$PWD/src:$PWD/scripts"
```

**立即重绘四张图（无需下载模型或数据）：**

```bash
python scripts/replot_results.py
```

输出固定为 `results/figures/figure7a.png`、`figure7b.png`、`figure8a.png`、`figure8b.png`。原始图和数据报告已随项目保存。

**重新运行实验前准备：** 用下载入口取得 CLIP、CLIPSeg、Qwen、SegFormer 并生成模型锁文件；接收端权重需按下述训练入口生成，或放在配置所指路径。Cityscapes、COCO/FSS 和 ImageNet 数据放在配置所指目录；从 LimitNet 官方项目取得代码及权重，再通过 `scripts/convert_limitnet_official.py` 转换。数据和权重均不包含在本项目压缩包中。ImageNet val 压缩包及标签需有合法访问来源。

```bash
python scripts/download_models.py
python scripts/prepare_figure7_classifier.py download
python scripts/prepare_imagenet_val_tar.py --tar data/imagenet_mirror/ILSVRC2012_img_val.tar --labels data/imagenet_mirror/imagenet_2012_validation_synset_labels.txt
python scripts/run_figure7_classification.py validate --val-root data/imagenet_val
```

首次训练恢复网络时，还需按 `configs/image_autonomous_driving.yaml` 放置 COCO/FSS 图像与官方 FSS 类别划分，再依次运行：

```bash
python scripts/prepare_pilot_fold0.py
python scripts/prepare_pilot_bf16.py --epochs 15
python scripts/build_receiver_kb.py
python scripts/cache_receiver_tokens.py
torchrun --standalone --nproc_per_node=4 scripts/train_pilot_15ep_ddp.py
```

**图 8(a) 试跑入口：** 先在 Cityscapes train 上运行 `scripts/tune_gt_30kb_sender.py`，得到冻结的发送端配置；再用相同预算与随机丢包设置评测 val。将下面两个占位路径替换为实际文件。

```bash
python scripts/run_figure8a_gt_random_val.py --config configs/tx_stage12_qwen4b.yaml --task all --budget-bytes 30000 --max-images 100 --sender-profile-config <frozen_profiles.json> --limitnet-checkpoint <limitnet_cityscapes.pt> --device cuda:0 --output artifacts/figure8a_gt_random_100
```

**图 7 和图 8(b) 入口：** 先运行 `scripts/prepare_figure7_bins.py` 准备分类任务语义桶，`scripts/run_figure7_classification.py intact` 产生原图分类结果。图 7 用 `scripts/run_figure7_pilot_queue.py` 跑各点，再用 `scripts/merge_figure7_pilot_curve.py` 汇总；图 8(b) 用 `scripts/run_figure8b_pilot_sweep.py` 跑 64–512 像素，用 `scripts/run_figure8b_kdc_1024_pilot.py` 和 `scripts/run_figure8b_limitnet_1024_pilot.py` 跑 1024 像素，再用 `scripts/merge_figure8b_pilot_sweep.py` 汇总。每个入口的必需路径和参数可运行 `python scripts/<脚本名>.py --help` 查看。重新评测会将中间文件写到 `artifacts/`，不会覆盖上述四张已归档展示图。

