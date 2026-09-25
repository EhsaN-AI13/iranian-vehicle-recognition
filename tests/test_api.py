from fastapi.testclient import TestClient

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

    with open(
        "test_images/car5.jpg",
        "rb",
    ) as image:

        response = client.post(
            "/predict",
            files={
                "file": (
                    "car5.jpg",
                    image,
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