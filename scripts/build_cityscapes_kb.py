"""独立构建只包含 Cityscapes train 标注区域和 19 类词表的 PosteriorKB。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from replay.core.config import load_config, require_device
from replay.knowledge.build import build_cityscapes_only_kb, cityscapes_only_inputs
from replay.knowledge.encoders import SharedKB


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/image_autonomous_driving.yaml",
                        help="项目根目录下的配置文件；参考清单必须仍指向指定 Cityscapes train 清单")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    destination = ROOT / "artifacts/posterior_kb_cityscapes_train"
    # 先核验所有来源和输出路径，再载入需要 GPU 的 CLIP 权重。
    rows, entries, source_audit = cityscapes_only_inputs(config, destination)
    shared = SharedKB.from_config(config, require_device(config.values["device"]))
    audit = build_cityscapes_only_kb(config, shared, destination, rows, entries, source_audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
