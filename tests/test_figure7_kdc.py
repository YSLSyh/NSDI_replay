"""图 7 KDC：真实线字节、无标签发送端与缺失依赖。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

import numpy as np
from PIL import Image

from replay.core.config import ReplayConfig
from replay.evaluation.figure7_kdc import (
    Figure7KdcSpec, KdcClassificationCodec, KdcWireBytes,
    _classification_config, _missing_mask_from_control,
    missing_kdc_classification_assets,
)
from replay.transmission.packet_wire import receive_packetized_frame


class FakeImportance:
    def map(self, image, bins, return_scores=False):
        rows, cols = (image.height + 15) // 16, (image.width + 15) // 16
        scores = np.ones((rows, cols), np.float32) / (rows * cols)
        selected = np.ones((rows, cols), dtype=bool)
        heat = np.ones((image.height, image.width), np.float32)
        return heat, selected, scores


class Figure7KdcTests(unittest.TestCase):
    _TEMP_PARENT = Path(__file__).resolve().parents[1] / "artifacts"

    def _codec(self, *, restorer=None):
        config = ReplayConfig(Path("."), {
            "task": {"description": "classify the primary object in an image"},
            "image": {"patch_size": 16},
            "models": {"clip": {"repo": "openai/clip-vit-base-patch16",
                                "revision": "test-revision"}},
            "importance": {
                "dct_high_frequency_cutoff": 2,
                "prequant_luma_strength": 2.0,
                "prequant_chroma_strength": 2.0,
                "prequant_gamma": 1.0,
                "importance_ratio_low": 1.0,
                "importance_ratio_high": 4.0,
            },
            "knowledge": {"rx_top_k": 4, "session_merge_cosine": 0.95},
        })
        return KdcClassificationCodec(
            config=config, bins=["person", "animal", "vehicle", "plant"],
            shared=object(), importance=FakeImportance(), kb=object(),
            receiver=object(), device="cpu", fixed_loss_budget_bytes=30_000,
            restorer=restorer)

    def test_budget_counts_all_control_and_packet_bytes(self) -> None:
        rng = np.random.default_rng(7)
        image = Image.fromarray(rng.integers(0, 256, (64, 64, 3), np.uint8))
        codec = self._codec()
        payload, metadata = codec.encode(image, budget_bytes=12_000)
        self.assertIsInstance(payload, KdcWireBytes)
        self.assertEqual(len(payload), metadata["wire_bytes"])
        self.assertEqual(payload.frame.sent_bytes, metadata["wire_bytes"])
        self.assertLessEqual(metadata["wire_bytes"], 12_000)
        self.assertEqual(metadata["loss_packet_count"], len(payload.frame.packets))
        self.assertGreater(metadata["control_bytes"], metadata["frame_header_bytes"])
        result = codec.decode(payload, loss_flags=(False,) * len(payload.frame.packets),
                              received_prefix_bytes=None)
        self.assertIsNotNone(result)
        self.assertEqual(result.size, image.size)

    def test_missing_mask_is_reconstructed_from_received_control(self) -> None:
        rng = np.random.default_rng(8)
        image = Image.fromarray(rng.integers(0, 256, (96, 96, 3), np.uint8))
        payload, _ = self._codec().encode(image, budget_bytes=20_000)
        dropped = [True] + [False] * (len(payload.frame.packets) - 1)
        received = receive_packetized_frame(
            payload.frame.control,
            [packet for packet, lost in zip(payload.frame.packets, dropped)
             if not lost])
        mask = _missing_mask_from_control(
            payload.frame.control, received.lost_intervals,
            width=image.width, height=image.height)
        self.assertEqual(mask.shape, (96, 96))
        self.assertGreater(float(mask.sum()), 0)

    def test_missing_assets_are_reported_instead_of_fabricated(self) -> None:
        spec = Figure7KdcSpec(
            config_path=Path(__file__).resolve().parents[1] / "configs/tx_stage12_qwen4b.yaml",
            task_description="classify the primary object in an image",
            bins_cache=Path("artifacts/absent_figure7_bins.json"),
            kb_path=Path("artifacts/absent_figure7_kb"),
            receiver_checkpoint=Path("artifacts/absent_figure7_receiver.pt"),
            fixed_loss_budget_bytes=30_000)
        issues = missing_kdc_classification_assets(spec)
        self.assertTrue(any("语义桶" in issue for issue in issues))
        self.assertTrue(any("PosteriorKB" in issue for issue in issues))
        self.assertTrue(any("图像恢复权重" in issue for issue in issues))
        with self.assertRaisesRegex(FileNotFoundError, "依赖不完整"):
            KdcClassificationCodec.from_spec(spec, "cpu")
        self.assertFalse(spec.update_session)

    def test_classification_task_discards_cityscapes_bin_constraints(self) -> None:
        spec = Figure7KdcSpec(
            config_path=Path(__file__).resolve().parents[1] / "configs/tx_stage12_qwen4b.yaml",
            task_description="classify the primary object in an image",
            bins_cache=Path("artifacts/figure7_object_bins.json"),
            kb_path=Path("artifacts/posterior_kb_pilot_fold0"),
            receiver_checkpoint=Path("artifacts/receiver.pt"),
            fixed_loss_budget_bytes=30_000)
        config = _classification_config(spec)
        task = config.section("task")
        self.assertEqual(task["description"], spec.task_description)
        for key in ("required_bins", "allowed_bins", "bin_aliases", "taxonomy"):
            self.assertNotIn(key, task)

    def test_check_rejects_old_phone_synonym_cache(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temporary:
            cache = Path(temporary) / "old_bins.json"
            task = "Recognize the main object in a photograph for ImageNet-1K classification"
            cache.write_text(json.dumps({
                "task": task,
                "bins": ["cell phone", "phone", "smartphone", "mobile device"]}),
                encoding="utf-8")
            spec = Figure7KdcSpec(
                config_path=Path(__file__).resolve().parents[1] / "configs/tx_stage12_qwen4b.yaml",
                task_description=task, bins_cache=cache,
                kb_path=Path("artifacts/posterior_kb_pilot_fold0"),
                receiver_checkpoint=Path("artifacts/receiver.pt"),
                fixed_loss_budget_bytes=30_000)
            issues = missing_kdc_classification_assets(spec)
            self.assertTrue(any("同义过滤审计" in issue for issue in issues))

    def test_check_accepts_selected_audit_attempt_even_if_later_attempt_failed(self) -> None:
        with TemporaryDirectory(dir=self._TEMP_PARENT) as temporary:
            cache = Path(temporary) / "reaudited_bins.json"
            task = "Recognize main objects across broad image classes"
            bins = ["animal", "vehicle", "furniture", "food"]
            cache.write_text(json.dumps({
                "task": task, "bins": bins,
                "semantic_distinct_count": 4,
                "semantic_diversity_verified": True,
                "broad_family_verified": True,
                "selection_audit": {"selected_attempt_index": 0, "attempts": [
                    {"bins": bins, "accepted": True,
                     "family_audit": {"accepted": True}},
                    {"bins": ["animal", "vehicle"], "accepted": False,
                     "family_audit": {"accepted": False}}]},
            }), encoding="utf-8")
            spec = Figure7KdcSpec(
                config_path=Path(__file__).resolve().parents[1] / "configs/tx_stage12_qwen4b.yaml",
                task_description=task, bins_cache=cache,
                kb_path=Path("artifacts/posterior_kb_pilot_fold0"),
                receiver_checkpoint=Path("artifacts/receiver.pt"),
                fixed_loss_budget_bytes=30_000)
            issues = missing_kdc_classification_assets(spec)
            self.assertFalse(any("同义过滤审计" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
