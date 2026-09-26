from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from api import app


client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_predict():
    image = Image.new("RGB", (224, 224), color="white")

    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.jpg",
                buffer,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "filename" in data
    assert "predictions" in data
    assert len(data["predictions"]) == 3
    assert "class" in data["predictions"][0]
    assert "confidence" in data["predictions"][0]
