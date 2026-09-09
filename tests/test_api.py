from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()
        assert "documents_indexed" in response.json()


def test_ask_validation_empty_query():
    # التحقق من أن الاستعلام الفارغ يعيد 422
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_valid_query():
    with TestClient(app) as test_client:
        response = test_client.post("/ask", json={"question": "عقد البيع"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
