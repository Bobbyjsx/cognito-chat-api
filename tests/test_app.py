from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    print("Health check passed.")


def test_chat():
    response = client.post("/agent/chat", json={"message": "Hello! Just testing the connection."})
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "response" in data
    print("Chat check passed. Agent responded:", data["response"])


if __name__ == "__main__":
    test_health()
    test_chat()
