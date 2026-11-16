"""原图分类批处理与原有逐图预测保持同样的顺序和类别。"""

import unittest

from PIL import Image
import torch

from replay.evaluation.figure7_classification import Classifier


class _TinyClassifier(torch.nn.Module):
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        labels = batch[:, 0, 0, 0].long()
        logits = torch.zeros((len(batch), 1_000), dtype=torch.float32)
        logits.scatter_(1, labels[:, None], 1.0)
        return logits


class IntactBatchTests(unittest.TestCase):
    def test_batch_matches_single_image_predictions(self) -> None:
        classifier = Classifier.__new__(Classifier)
        classifier.model = _TinyClassifier().eval()
        classifier._preprocess = lambda image: torch.tensor(
            [[[[image.getpixel((0, 0))[0]]]]], dtype=torch.float32)
        images = [Image.new("RGB", (1, 1), (value, 0, 0))
                  for value in (7, 23, 3, 19)]
        expected = [classifier.predict(image) for image in images]
        self.assertEqual(classifier.predict_batch(images), expected)
        self.assertEqual(classifier.predict_batch(images[:1]), expected[:1])

    def test_empty_batch_is_rejected(self) -> None:
        classifier = Classifier.__new__(Classifier)
        with self.assertRaisesRegex(ValueError, "不能为空"):
            classifier.predict_batch([])


if __name__ == "__main__":
    unittest.main()
