# Credits: Built on open-source libraries and papers acknowledged in README.md citations.

"""Tests for the inference predictor."""

import torch
from PIL import Image

from src.inference.predictor import ImageTranslator
from src.models.generators import UNetGenerator


class TestImageTranslator:
    def test_predict_returns_pil(self):
        gen = UNetGenerator(in_channels=3, out_channels=3, num_downs=5, base_filters=16)
        translator = ImageTranslator(gen, device="cpu", image_size=32)
        img = Image.new("RGB", (64, 64))
        result = translator.predict(img)
        assert isinstance(result, Image.Image)

    def test_predict_batch(self):
        gen = UNetGenerator(in_channels=3, out_channels=3, num_downs=5, base_filters=16)
        translator = ImageTranslator(gen, device="cpu", image_size=32)
        imgs = [Image.new("RGB", (64, 64)) for _ in range(3)]
        results = translator.predict_batch(imgs)
        assert len(results) == 3
        assert all(isinstance(r, Image.Image) for r in results)

    def test_predict_file(self, tmp_path):
        gen = UNetGenerator(in_channels=3, out_channels=3, num_downs=5, base_filters=16)
        translator = ImageTranslator(gen, device="cpu", image_size=32)
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        Image.new("RGB", (64, 64)).save(input_path)
        translator.predict_file(input_path, output_path)
        assert output_path.exists()
        result = Image.open(output_path)
        assert result.mode == "RGB"
