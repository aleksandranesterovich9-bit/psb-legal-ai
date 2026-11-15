from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.document_loader import load_documents_from_zip
from solution import InMemoryVectorIndex, HybridRetriever


class FakeEmbeddingClient:
    def embed(self, texts):
        vector = [1.0] + [0.0] * 1023
        return [vector[:] for _ in texts]


class EmptyBM25:
    def search(self, **kwargs):
        return []


class MustNotRunVector:
    def search(self, **kwargs):
        raise AssertionError("Vector search must not bypass the BM25 evidence gate")


def test_official_vector_index_respects_historical_revision_dates():
    root = Path(__file__).resolve().parents[1]
    documents = load_documents_from_zip(
        root / "data" / "documents" / "knowledge-base.zip"
    )
    chunks = chunk_documents(documents)

    index = InMemoryVectorIndex(FakeEmbeddingClient())
    index.build(chunks)
    results = index.search(
        "Договор заключали в августе 2024 года. Какой регламент действовал?",
        top_k=100,
    )

    revision_titles = {
        result.chunk.title
        for result in results
        if "Регламент согласования договоров" in result.chunk.title
    }
    assert any("ред 2023" in title for title in revision_titles)
    assert not any("ред 2025" in title for title in revision_titles)


def test_hybrid_search_does_not_return_vector_only_noise():
    retriever = HybridRetriever(
        bm25_retriever=EmptyBM25(),
        vector_index=MustNotRunVector(),
    )

    assert retriever.search("вопрос без доказательств") == []
