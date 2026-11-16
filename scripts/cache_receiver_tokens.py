"""为接收端训练缓存原尺寸 CLIP patch token（FP32）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay.core.config import load_config, require_device
from replay.data.token_cache import cache_tokens
from replay.knowledge.encoders import SharedKB


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/pilot_fold0_bf16_15_epochs.yaml"))
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    shared = SharedKB.from_config(config, require_device(config.values["device"]))
    # 缓存按模型修订与 patch 投影方式分目录；已有缓存会先校验元信息。
    print(json.dumps(cache_tokens(config, shared), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
