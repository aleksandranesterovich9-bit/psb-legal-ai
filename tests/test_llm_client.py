
import pytest
import httpx

from app.llm_client import AIClient


@pytest.fixture
def client():
    client = AIClient()

    client.base_url = "https://test-api.local"
    client.llm_key = "test-key"
    client.embed_key = "test-key"
    client.rerank_key = "test-key"

    def mock_api(request):
        if request.url.path == "/embeddings":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": 1, "embedding": [0, 2]},
                        {"index": 0, "embedding": [3, 4]},
                    ]
                },
            )

        if request.url.path == "/rerank":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1},
                        {"index": 0},
                    ]
                },
            )

        if request.url.path == "/chat/completions":
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "Договор согласовывается."
                            }
                        }
                    ]
                },
            )

        return httpx.Response(404)

    client.http.close()
    client.http = httpx.Client(
        transport=httpx.MockTransport(mock_api)
    )

    yield client

    client.close()


def test_embed(client):
    vectors = client.embed(["Текст 1", "Текст 2"])

    assert len(vectors) == 2

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.0, 1.0])


def test_rerank(client):
    indices = client.rerank(
        query="Как согласовать договор?",
        documents=["Текст 1", "Текст 2"],
        top_n=2,
    )

    assert indices == [1, 0]


def test_chat(client):
    answer = client.chat(
        messages=[
            {
                "role": "user",
                "content": "Как согласовать договор?",
            }
        ]
    )

    assert answer == "Договор согласовывается."
