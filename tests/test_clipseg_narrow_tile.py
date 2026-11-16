"""CLIPSeg 图块通道顺序回归测试。"""

import sys
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from replay.transmission.importance import _clipseg_inputs  # noqa: E402


class _ImageProcessor:
    def __call__(self, *, images, return_tensors, input_data_format):
        assert input_data_format == "channels_last"
        assert return_tensors == "pt"
        assert all(image.mode == "RGB" for image in images)
        return {"pixel_values": np.stack([np.asarray(image) for image in images])}


class _Processor:
    image_processor = _ImageProcessor()

    def __call__(self, *, text, padding, return_tensors):
        assert padding and return_tensors == "pt"
        return {"input_ids": list(text)}


class ClipSegNarrowTileTests(unittest.TestCase):
    def test_rgb_edge_tiles_keep_three_channels(self):
        for shape in ((1, 352, 3), (352, 1, 3), (352, 352, 3)):
            with self.subTest(shape=shape):
                tile = Image.fromarray(np.zeros(shape, dtype=np.uint8), "RGB")
                inputs = _clipseg_inputs(_Processor(), tile, ["car", "person"])
                self.assertEqual(inputs["input_ids"], ["car", "person"])
                self.assertEqual(inputs["pixel_values"].shape, (2, *shape))


if __name__ == "__main__":
    unittest.main()
