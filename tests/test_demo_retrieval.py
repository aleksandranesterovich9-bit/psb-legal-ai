
import csv
from pathlib import Path

import pytest

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Retriever


@pytest.fixture(scope="module")
def retriever():
    """
    Создаёт поисковый индекс один раз
    для всех 18 тестовых вопросов.
    """

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


def load_questions():
    """Загружает официальные вопросы из CSV-файла."""

    project_root = Path(__file__).resolve().parents[1]

    questions_path = (
        project_root
        / "data"
        / "demo_questions.csv"
    )

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Файл с вопросами не найден: {questions_path}"
        )

    with open(
        questions_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        return list(reader)


def test_all_demo_questions(retriever):
    """
    Проверяет поиск BM25 на всех официальных вопросах.
    Выводит пять наиболее подходящих фрагментов
    для каждого вопроса.
    """

    questions = load_questions()

    assert len(questions) == 18, (
        f"Ожидалось 18 вопросов, найдено: {len(questions)}"
    )

    for item in questions:

        question_id = item["id"]
        question = item["question"]

        results = retriever.search(
            query=question,
            top_k=5
        )

        print("\n" + "=" * 80)
        print(f"ВОПРОС №{question_id}")
        print(question)

        if not results:
            print("\nНичего не найдено.")
            continue

        for position, result in enumerate(results, start=1):

            chunk = result.chunk

            print(f"\n--- РЕЗУЛЬТАТ {position} ---")
            print(f"Документ: {chunk.title}")
            print(f"Файл: {chunk.filename}")
            print(f"Статья: {chunk.article}")
            print(f"Раздел: {chunk.section}")
            print(f"Пункт: {chunk.clause}")
            print(f"Оценка BM25: {result.score:.4f}")
            print(f"Текст: {chunk.text[:700]}")
