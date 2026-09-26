from pathlib import Path

from PIL import Image

from predictor import VehiclePredictor


def test_predictor_inference(tmp_path):
    image = Image.new("RGB", (224, 224), color="white")

    image_path = Path(tmp_path) / "test.jpg"
    image.save(image_path, format="JPEG")

    predictor = VehiclePredictor()

    results = predictor.predict(str(image_path))

    assert len(results) == 3
    assert all("class" in result for result in results)
    assert all("confidence" in result for result in results)
