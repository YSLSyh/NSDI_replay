"""准备图 7 的官方 Torchvision ViT-L/16 SWAG E2E 分类权重。"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


WEIGHT_NAME = "vit_l_16_swag-4f3808c9.pth"
WEIGHT_URL = f"https://download.pytorch.org/models/{WEIGHT_NAME}"
SHA256_PREFIX = "4f3808c9"
MODEL_SOURCE = "https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py"


def checksum(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, *, load_model: bool = False) -> dict[str, object]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = checksum(path)
    if not digest.startswith(SHA256_PREFIX):
        raise ValueError(f"官方权重 SHA-256 前缀错误：{digest}")
    result: dict[str, object] = {
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": digest, "source": WEIGHT_URL,
        "model": "torchvision ViT-L/16 SWAG E2E", "image_size": 512,
        "official_imagenet1k_top1": 0.88064,
        "paper_specified_classifier": False,
    }
    if load_model:
        # 512 像素模型的 positional embedding 与默认 224 模型不同。
        from replay.evaluation.figure7_classification import Classifier

        classifier = Classifier(
            backend="torchvision_vit_l_16_swag_e2e",
            weights_path=path, device="cpu")
        result["strict_model_load_valid"] = True
        del classifier
    return result


def download(path: Path) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return verify(path)
    part = path.with_name(path.name + ".part")
    request = Request(WEIGHT_URL, headers={"User-Agent": "replay-figure7-assets/1.0"})
    try:
        with urlopen(request, timeout=120) as response, part.open("wb") as output:
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                output.write(chunk)
        result = verify(part)
        os.replace(part, path)
        result["path"] = str(path.resolve())
        return result
    finally:
        if part.exists():
            part.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("info", "download", "verify"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/classification"))
    parser.add_argument("--load-model", action="store_true",
                        help="verify 时额外验证 512 像素 ViT-L 严格加载")
    args = parser.parse_args()
    path = args.output_dir / WEIGHT_NAME
    if args.command == "info":
        result = {
            "weight_url": WEIGHT_URL, "sha256_prefix": SHA256_PREFIX,
            "model_source": MODEL_SOURCE, "official_imagenet1k_top1": 0.88064,
            "input_size": [512, 512], "classifier_is_reproduction_choice": True,
        }
    elif args.command == "download":
        result = download(path)
    else:
        result = verify(path, load_model=args.load_model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
