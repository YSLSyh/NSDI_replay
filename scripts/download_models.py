"""按配置下载所需 Hugging Face 模型，并写入可核验的模型锁文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from replay.core.assets import ALLOW, download_models
from replay.core.config import load_config


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("configs/tx_stage12_qwen4b.yaml"))
    parser.add_argument("models", nargs="*", choices=sorted(ALLOW),
                        help="默认下载配置所需的全部模型")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_config(config_path)
    # 只通过配置中固定的仓库与修订下载；锁文件供后续推理入口核验。
    result = download_models(config, args.models or sorted(ALLOW))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
