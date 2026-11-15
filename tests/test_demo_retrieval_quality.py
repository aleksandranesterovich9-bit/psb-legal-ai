import csv
from pathlib import Path

import pytest

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Retriever


@pytest.fixture(scope="module")
def retriever():
    root = Path(__file__).resolve().parents[1]
    documents = load_documents_from_zip(
        root / "data" / "documents" / "knowledge-base.zip"
    )
    return BM25Retriever(chunk_documents(documents))


def load_questions():
    root = Path(__file__).resolve().parents[1]
    with (root / "data" / "demo_questions.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        return {row["id"]: row["question"] for row in csv.DictReader(file)}


@pytest.mark.parametrize(
    "question_id, expected_title_part, expected_clause",
    [
        ("1", "Договор аренды А-08-2024", "3.1"),
        ("2", "ТК_ст70", None),
        ("5", "Регламент согласования договоров ред 2023", "2.4"),
        ("6", "Договор поставки П-17-2024", "6.1"),
        ("7", "Договор аренды А-08-2024", "4.3"),
        ("8", "Договор поставки П-17-2024", "6.2"),
        ("9", "Договор поставки ПТ-05-2023", "4.2"),
        ("10", "Регламент согласования договоров ред 2025", "1.3"),
        ("11", "Служебная записка об архиве", None),
        ("12", "ТК_ст70", None),
        ("13", "152ФЗ_ст9", None),
        ("14", "ТК_ст59", None),
        ("15", "Договор аренды А-08-2024", "6.3"),
        ("16", "Договор поставки П-17-2024", "2.1"),
        ("17", "Договор поставки ПТ-05-2023", "6.1"),
        ("18", "Регламент согласования договоров ред 2025", "3.2"),
    ],
)
def test_expected_source_is_first(retriever, question_id, expected_title_part, expected_clause):
    question = load_questions()[question_id]
    results = retriever.search(question, top_k=5)

    assert results, f"Для вопроса {question_id} не найдено источников"
    first = results[0].chunk
    assert expected_title_part in first.title
    if expected_clause is not None:
        assert first.clause == expected_clause


def test_missing_maternity_benefit_information_abstains(retriever):
    question = load_questions()["3"]
    assert retriever.search(question, top_k=5) == []
