import time

from fastapi.testclient import TestClient

from app.api import server


def test_web_flow_without_api_keys(monkeypatch):
    for name in ("LLM_BASE_URL", "BASE_URL", "OPENAI_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "_load_env_file", lambda: None)
    server.STATE.update({"assistant": None, "error": None, "indexing": True})
    with TestClient(server.app) as client:
        for _ in range(100):
            health = client.get("/api/health").json()
            if health["status"] != "starting":
                break
            time.sleep(0.05)
        assert health["status"] == "ok" and health["documents"] == 21 and health["llm"] is False

        assert "PSB Legal AI" in client.get("/").text
        assert len(client.get("/api/examples").json()) >= 5
        docs = client.get("/api/documents").json()
        assert len(docs) == 21 and any(d["repealed"] for d in docs)

        reply = client.post("/api/chat", json={"messages": [{"role": "user", "content": "Какой гарантийный срок по договору П-17/2024?"}]}).json()
        assert reply["status"] == "extractive" and "24" in reply["answer"]
        citation = reply["citations"][0]
        assert citation["document"]["filename"].endswith("Альфа-Трейд.txt") and citation["quote"]

        document = client.get(f"/api/documents/{citation['document']['id']}").json()
        assert any(chunk["chunk_id"] == citation["chunk_id"] for chunk in document["chunks"])

        dialogue = [
            {"role": "user", "content": "Мы арендуем склад у «Северного Склада»."},
            {"role": "assistant", "content": "Что именно уточнить?"},
            {"role": "user", "content": "Как возвращать помещение при прекращении договора?"},
        ]
        reply = client.post("/api/chat", json={"messages": dialogue}).json()
        assert "акту" in reply["answer"]

        assert client.post("/api/chat", json={"messages": []}).status_code == 422
        assert client.get("/api/documents/nope").status_code == 404
