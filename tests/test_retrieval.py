
from pathlib import Path

import pytest

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Retriever


@pytest.fixture(scope="module")
def retriever():
    """Создаёт поисковый индекс один раз для всех тестов."""

    project_root = Path(__file__).resolve().parents[1]

    archive_path = (
        project_root
        / "data"
        / "documents"
        / "knowledge-base.zip"
    )

    documents = load_documents_from_zip(archive_path)
    chunks = chunk_documents(documents)

    return BM25Retriever(chunks)


def test_search_lease_termination(retriever):
    results = retriever.search(
        query="Как расторгнуть договор аренды?",
        top_k=5
    )

    assert len(results) > 0

    print("\nПОИСК: расторжение договора аренды")

    for result in results:
        print("\n" + "=" * 60)
        print(f"Документ: {result.chunk.title}")
        print(f"Пункт: {result.chunk.clause}")
        print(f"Оценка BM25: {result.score:.4f}")
        print(result.chunk.text[:500])

    assert any(
        result.chunk.clause == "6.2"
        for result in results
    )


def test_search_personal_data(retriever):
    results = retriever.search(
        query="Согласие субъекта персональных данных на обработку",
        top_k=5,
        category="law"
    )

    assert len(results) > 0

    print("\nПОИСК: согласие на обработку персональных данных")

    for result in results:
        print("\n" + "=" * 60)
        print(f"Документ: {result.chunk.title}")
        print(f"Статья: {result.chunk.article}")
        print(f"Часть: {result.chunk.part}")
        print(f"Оценка BM25: {result.score:.4f}")
        print(result.chunk.text[:500])

    assert any(
        result.chunk.article
        and "Статья 9" in result.chunk.article
        for result in results
    )


def test_search_empty_query(retriever):
    results = retriever.search("")

    assert results == []


def test_search_category_filter(retriever):
    results = retriever.search(
        query="Договор аренды помещения",
        top_k=10,
        category="company_document"
    )

    assert len(results) > 0

    for result in results:
        assert result.chunk.category == "company_document"
