"""验证官方校验失败会停下、tar 成员安全和标签按编号对应。"""

from __future__ import annotations

from hashlib import md5
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_imagenet_val_tar.py"
WORK_DIR = SCRIPT.parents[1]
spec = spec_from_file_location("prepare_imagenet_val_tar", SCRIPT)
assert spec is not None and spec.loader is not None
preparer = module_from_spec(spec)
spec.loader.exec_module(preparer)


def make_tar(path: Path, files: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w") as archive:
        for name, raw in files:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            archive.addfile(info, BytesIO(raw))


class PrepareImageNetValTarTests(unittest.TestCase):
    def test_archive_md5_must_match_before_extraction(self) -> None:
        with TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "val.tar"
            path.write_bytes(b"not the official ImageNet archive")
            with self.assertRaisesRegex(ValueError, "拒绝解压"):
                preparer.verify_official_archive(path)
            self.assertEqual(
                preparer.verify_official_archive(path, expected_md5=md5(path.read_bytes()).hexdigest()),
                md5(path.read_bytes()).hexdigest(),
            )

    def test_tar_rejects_traversal_and_non_file_members(self) -> None:
        with TemporaryDirectory(dir=WORK_DIR) as directory:
            path = Path(directory) / "bad.tar"
            make_tar(path, [("../ILSVRC2012_val_00000001.JPEG", b"\xff\xd8x")])
            with tarfile.open(path) as archive:
                with self.assertRaisesRegex(ValueError, "非官方普通 JPEG"):
                    preparer.archive_members(archive, expected_images=1)

    def test_labels_are_paired_with_filename_not_tar_order(self) -> None:
        with TemporaryDirectory(dir=WORK_DIR) as directory:
            root = Path(directory)
            names = [f"ILSVRC2012_val_{index:08d}.JPEG" for index in range(1, 5)]
            # 故意打乱 tar 顺序，防止把 archive 顺序误当验证图编号。
            path = root / "val.tar"
            make_tar(path, [(names[index - 1], b"\xff\xd8" + bytes([index]))
                            for index in (4, 1, 3, 2)])
            labels = preparer.validation_labels(
                b"n00000001\nn00000002\nn00000001\nn00000002\n",
                expected_images=4, expected_classes=2, expected_per_class=2,
            )
            output = root / "output"
            with tarfile.open(path) as archive:
                members = preparer.archive_members(archive, expected_images=4)
                preparer.extract_archive(archive, members, labels, output)
            for index, label in enumerate(labels, 1):
                self.assertEqual((output / label / names[index - 1]).read_bytes(),
                                 b"\xff\xd8" + bytes([index]))


if __name__ == "__main__":
    unittest.main()
