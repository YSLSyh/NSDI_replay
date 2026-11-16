"""核对 ImageNet-1K 官方验证集 tar，再按官方 WNID 标签整理 50,000 张图。

镜像仅作为传输渠道：只有 tar 的 MD5 与 torchvision 公布的官方校验值
完全相同，才读取并整理其中的图像。原始文件名和 JPEG 字节保持不变。
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import md5, sha256
import json
import os
from pathlib import Path
import re
import sys
import tarfile
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_VAL_MD5 = "29b22e2961454d5413ddabcf34fc5622"
LABELS_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_2012_validation_synset_labels.txt"
)
MIRROR_URL = (
    "https://huggingface.co/datasets/presencesw/ILSVRC2012/resolve/main/"
    "ILSVRC2012_img_val.tar"
)
EXPECTED_IMAGES = 50_000
EXPECTED_CLASSES = 1_000
EXPECTED_PER_CLASS = 50
NAME = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG\Z")
WNID = re.compile(r"n\d{8}\Z")


def project_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if path == PROJECT_ROOT or not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"仅允许在 replay 项目目录内操作：{path}")
    return path


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    digest = md5() if algorithm == "md5" else sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_official_archive(path: Path, *, expected_md5: str = OFFICIAL_VAL_MD5) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"验证集 tar 不存在：{path}")
    actual = file_digest(path, "md5")
    if actual != expected_md5:
        raise ValueError(
            f"验证集 tar 与官方 MD5 不一致：{actual} != {expected_md5}；拒绝解压"
        )
    return actual


def validation_labels(raw: bytes, *, expected_images: int = EXPECTED_IMAGES,
                      expected_classes: int = EXPECTED_CLASSES,
                      expected_per_class: int = EXPECTED_PER_CLASS) -> list[str]:
    try:
        labels = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("验证集 WNID 标签文件不是 ASCII 文本") from exc
    if len(labels) != expected_images or any(not WNID.fullmatch(label) for label in labels):
        raise ValueError("验证集 WNID 标签文件的行数或格式错误")
    counts = Counter(labels)
    if len(counts) != expected_classes or any(
        count != expected_per_class for count in counts.values()
    ):
        raise ValueError("验证集标签应为 1,000 类、每类 50 张")
    return labels


def archive_members(archive: tarfile.TarFile, *,
                    expected_images: int = EXPECTED_IMAGES) -> dict[int, tarfile.TarInfo]:
    """先完整审计 tar 成员，再读取任意图像；禁止路径和符号链接。"""
    by_index: dict[int, tarfile.TarInfo] = {}
    for member in archive:
        match = NAME.fullmatch(member.name)
        if not member.isfile() or member.size <= 0 or match is None:
            raise ValueError(f"验证集 tar 含非官方普通 JPEG 文件：{member.name}")
        index = int(match.group(1))
        if not 1 <= index <= expected_images or index in by_index:
            raise ValueError(f"验证集 tar 图像编号重复或越界：{member.name}")
        by_index[index] = member
    if len(by_index) != expected_images or set(by_index) != set(range(1, expected_images + 1)):
        raise ValueError(f"验证集 tar 应包含恰好 {expected_images} 张官方命名的图像")
    return by_index


def _load_labels(path: Path) -> tuple[list[str], str]:
    if not path.exists():
        # TensorFlow 官方文件逐行对应 ILSVRC2012_val_00000001..00050000。
        # 仅下载公开的标签文本；图像本身来自已核对 MD5 的 tar。
        with urlopen(LABELS_URL, timeout=30) as response:
            raw = response.read()
        validation_labels(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=".imagenet-labels-",
                                delete=False) as tmp:
            temporary = Path(tmp.name)
            try:
                tmp.write(raw)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            if path.exists():
                raise ValueError(f"标签文件在下载期间出现：{path}")
            temporary.rename(path)
        finally:
            temporary.unlink(missing_ok=True)
    raw = path.read_bytes()
    return validation_labels(raw), sha256(raw).hexdigest()


def _validate_output_root(root: Path, valid_wnids: set[str]) -> None:
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"输出路径不是普通目录：{root}")
    for class_dir in root.iterdir():
        if (class_dir.is_symlink() or not class_dir.is_dir() or
                class_dir.name not in valid_wnids):
            raise ValueError(f"输出路径含无关内容，拒绝修改：{class_dir}")
        for child in class_dir.iterdir():
            if (child.is_symlink() or not child.is_file() or
                    NAME.fullmatch(child.name) is None):
                raise ValueError(f"类别目录含无关内容，拒绝修改：{child}")


def _write_new_or_matching(path: Path, raw: bytes) -> None:
    if path.exists():
        if (path.is_file() and not path.is_symlink() and
                path.stat().st_size == len(raw) and
                sha256(path.read_bytes()).digest() == sha256(raw).digest()):
            return
        raise ValueError(f"已有验证图内容不同，拒绝覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".imagenet-part-",
                            delete=False) as tmp:
        temporary = Path(tmp.name)
        try:
            tmp.write(raw)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.exists():
            raise ValueError(f"写入时验证图已出现：{path}")
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def extract_archive(archive: tarfile.TarFile,
                    members: dict[int, tarfile.TarInfo],
                    labels: list[str], output_root: Path) -> None:
    _validate_output_root(output_root, set(labels))
    for index in range(1, len(labels) + 1):
        member = members[index]
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"无法读取 tar 成员：{member.name}")
        with source:
            raw = source.read()
        if len(raw) != member.size or not raw.startswith(b"\xff\xd8"):
            raise ValueError(f"tar 成员不是完整 JPEG：{member.name}")
        _write_new_or_matching(output_root / labels[index - 1] / member.name, raw)
        if index % 5_000 == 0:
            print(f"已整理 {index}/{len(labels)} 张验证图", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", default="data/imagenet_mirror/ILSVRC2012_img_val.tar")
    parser.add_argument("--labels", default="data/imagenet_mirror/imagenet_2012_validation_synset_labels.txt")
    parser.add_argument("--val-root", default="data/imagenet_val")
    parser.add_argument("--verify-only", action="store_true",
                        help="只核对官方 MD5 和 tar 结构，不解压")
    args = parser.parse_args(argv)
    archive_path = project_path(args.tar)
    labels_path = project_path(args.labels)
    output_root = project_path(args.val_root)
    if len({archive_path, labels_path, output_root}) != 3:
        parser.error("输入、标签和输出路径不得相同")

    tar_md5 = verify_official_archive(archive_path)
    print(f"tar 的官方 MD5 已核对：{tar_md5}", flush=True)
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive_members(archive)
        print(f"tar 结构已核对：{len(members)} 张官方验证图", flush=True)
        if args.verify_only:
            return 0
        labels, labels_sha256 = _load_labels(labels_path)
        extract_archive(archive, members, labels, output_root)

    # 与图 7 的 50,000 张、1,000 个 WNID 目录校验使用同一入口。
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from replay.evaluation.figure7_classification import validate_imagenet_val
    samples, manifest_sha256 = validate_imagenet_val(output_root)
    manifest = {
        "image_count": len(samples),
        "source": "third-party Hugging Face mirror, bitwise verified against official ImageNet tar MD5",
        "mirror_url": MIRROR_URL,
        "archive_name": archive_path.name,
        "archive_md5": tar_md5,
        "labels_url": LABELS_URL,
        "labels_sha256": labels_sha256,
        "class_order": "WNID lexicographic (ImageFolder convention)",
        "dataset_manifest_sha256": manifest_sha256,
    }
    manifest_path = output_root.parent / "imagenet_val_manifest.json"
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    _write_new_or_matching(manifest_path, encoded)
    print(f"验证集整理完成：{output_root}；来源记录：{manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
