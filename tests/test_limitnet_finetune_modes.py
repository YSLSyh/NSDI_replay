"""固定尺寸 LimitNet 基线不能误用原生微调或验证集权重。"""

import random
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from replay.evaluation.figure8a_packet import _validate_limitnet_checkpoint_metadata


class LimitNetFineTuneModeTest(unittest.TestCase):
    def test_checkpoint_modes(self) -> None:
        self.assertEqual(
            _validate_limitnet_checkpoint_metadata(
                "official224", {"source": "official_limitnet_imagenet_pretrained"}),
            "official_imagenet_pretrained")
        fine = {
            "source": "cityscapes_finetuned_from_official",
            "training_dataset": "Cityscapes train",
            "input_mode": "official224_full_frame",
            "codec_input_resolution": [224, 224],
            "pixel_rescaling": True,
            "epochs_completed": 2,
        }
        self.assertEqual(_validate_limitnet_checkpoint_metadata("official224", fine),
                         "cityscapes_train_finetuned_224")
        for broken in ({**fine, "training_dataset": "Cityscapes val"},
                       {**fine, "input_mode": "native_crop"},
                       {**fine, "epochs_completed": 0},
                       {**fine, "source": "cityscapes_finetuned_from_checkpoint"}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                _validate_limitnet_checkpoint_metadata("official224", broken)
        native = {**fine, "input_mode": "native_crop", "pixel_rescaling": False}
        self.assertEqual(_validate_limitnet_checkpoint_metadata("native_adapted", native),
                         "cityscapes_train_finetuned_native")
        with self.assertRaises(ValueError):
            _validate_limitnet_checkpoint_metadata("native_adapted", fine)

    def test_fixed_input_uses_full_frame_and_nearest_labels(self) -> None:
        # 本机 transformers 版本可能不匹配；这里只测纯图像预处理逻辑。
        fake = types.ModuleType("transformers")
        fake.AutoImageProcessor = object
        fake.SegformerForSemanticSegmentation = object
        with patch.dict(sys.modules, {"transformers": fake}):
            from replay.baselines.limitnet_train import (FineTuneOptions,
                                                         _official_phase3_keep_mask,
                                                         _training_sample)
        image = Image.new("RGB", (16, 8), (100, 150, 200))
        label = np.zeros((8, 16), dtype=np.uint8)
        label[:, :8] = 13
        label[:, 8:] = 8
        options = FineTuneOptions(Path("train"), Path("segmenter"), Path("output.pt"),
                                  input_mode="official224_full_frame")
        actual, target = _training_sample(image, label, options, random.Random(1))
        self.assertEqual(actual.size, (224, 224))
        self.assertEqual(target.shape, (224, 224))
        self.assertTrue(np.all(target[:, 0] == 13))
        self.assertTrue(np.all(target[:, -1] == 8))
        self.assertEqual(image.size, (16, 8))
        saliency = torch.tensor([[[[0.1, 0.3], [0.5, 0.9]]]])
        retained = _official_phase3_keep_mask(saliency, 0.25)
        self.assertEqual(retained.shape, (1, 12, 2, 2))
        self.assertTrue(torch.all((retained == 0) | (retained == 1)))
        self.assertGreater(int(retained.sum()), 0)
        self.assertLess(int(retained.sum()), retained.numel())
        with self.assertRaises(ValueError):
            _official_phase3_keep_mask(saliency, 0.0)


if __name__ == "__main__":
    unittest.main()
