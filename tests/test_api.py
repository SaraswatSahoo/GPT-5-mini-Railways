import os
import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health():
    r = client.get("/health")  # Check /health, not root
    assert r.status_code == 200
    # Accept any valid status string from your health check
    assert r.json()["status"] in ["healthy", "degraded", "unhealthy"]

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OPENAI_API_KEY")
@pytest.mark.skipif(not os.getenv("MONGODB_URI"), reason="No MONGODB_URI")
def test_ask_smoke():
    headers = {
        "Authorization": f"Bearer {os.getenv('API_KEY')}"  # Add authentication
    }
    r = client.post("/ask", json={"question":"Hello"}, headers=headers)
    assert r.status_code == 200
    assert "answer" in r.json()
