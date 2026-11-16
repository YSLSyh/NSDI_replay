"""为 COCO/FSS 图像恢复训练构建不含验证图的 PosteriorKB。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay.core.config import load_config, require_device
from replay.knowledge.build import build_kb
from replay.knowledge.encoders import SharedKB


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/pilot_fold0_bf16_15_epochs.yaml"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    destination = config.data_path("kb_path")
    if destination.exists():
        raise FileExistsError(f"接收端知识库已存在，拒绝覆盖：{destination}")
    shared = SharedKB.from_config(config, require_device(config.values["device"]))
    # 训练清单中已经限定 COCO 60 类与 FSS 非测试 760 类；不引入 Cityscapes val。
    kb = build_kb(config, shared, pretrain_only=True)
    print(json.dumps({"kb_path": str(destination), "records": len(kb.records)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
