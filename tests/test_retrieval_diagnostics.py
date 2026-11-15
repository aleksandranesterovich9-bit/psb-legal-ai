
import csv
from pathlib import Path

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import BM25Retriever, tokenize


def test_retrieval_diagnostics():
    project_root = Path(__file__).resolve().parents[1]

    documents = load_documents_from_zip(
        project_root / "data" / "documents" / "knowledge-base.zip"
    )
    chunks = chunk_documents(documents)
    retriever = BM25Retriever(chunks)

    with open(
        project_root / "data" / "demo_questions.csv",
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        questions = {
            row["id"]: row["question"]
            for row in csv.DictReader(file)
        }

    targets = {
        "9": {
            "title": "ПТ-05-2023 Альфа-Трейдинг",
            "clause": "4.2",
            "queries": [
                ("Полный вопрос", questions["9"]),
                ("Короткий запрос", "оплата техники авансом или после поставки"),
                (
                    "Запрос с юридическими терминами",
                    "предоплата по соглашению размер предоплаты счет поставщика"
                ),
            ],
        },
        "15": {
            "title": "А-08-2024 Северный Склад",
            "clause": "6.3",
            "queries": [
                ("Полный вопрос", questions["15"]),
                ("Короткий запрос", "А-08/2024 сдать ключи акт возврат помещения"),
                (
                    "Запрос с юридическими терминами",
                    "при прекращении договора вернуть помещение по акту"
                ),
            ],
        },
        "17": {
            "title": "ПТ-05-2023 Альфа-Трейдинг",
            "clause": "6.1",
            "queries": [
                ("Полный вопрос", questions["17"]),
                ("Короткий запрос", "Альфа-Трейдинг компьютеры на сколько гарантия"),
                (
                    "Запрос с юридическими терминами",
                    "гарантийный срок товар производитель гарантийный талон"
                ),
            ],
        },
    }

    print(f"\nВсего документов: {len(documents)}")
    print(f"Всего чанков: {len(chunks)}")

    for question_id, target in targets.items():
        print("\n" + "#" * 80)
        print(f"ВОПРОС №{question_id}")
        print(f"Целевой документ: {target['title']}")
        print(f"Целевой пункт: {target['clause']}")

        target_chunk = next(
            (
                chunk for chunk in chunks
                if target["title"] in chunk.title
                and chunk.clause == target["clause"]
            ),
            None
        )

        if target_chunk is None:
            print("Целевой чанк не найден!")
            continue

        print("\nТекст целевого чанка:")
        print(target_chunk.text)

        for query_name, query in target["queries"]:
            results = retriever.search(
                query=query,
                top_k=len(chunks)
            )

            target_rank = next(
                (
                    position
                    for position, result in enumerate(results, start=1)
                    if result.chunk.chunk_id == target_chunk.chunk_id
                ),
                None
            )

            target_result = next(
                (
                    result for result in results
                    if result.chunk.chunk_id == target_chunk.chunk_id
                ),
                None
            )

            query_tokens = set(tokenize(query))
            chunk_tokens = set(tokenize(target_chunk.text))
            common_tokens = sorted(query_tokens & chunk_tokens)

            print("\n" + "-" * 70)
            print(f"Вариант: {query_name}")
            print(f"Запрос: {query}")
            print(f"Позиция целевого пункта: {target_rank}")

            if target_result:
                print(f"Оценка целевого пункта: {target_result.score:.4f}")
            else:
                print("Оценка целевого пункта: 0 (не попал в выдачу)")

            print(f"Совпадающие токены: {common_tokens}")

            print("\nТОП-5 РЕЗУЛЬТАТОВ:")

            for position, result in enumerate(results[:5], start=1):
                chunk = result.chunk

                print(
                    f"{position}. {chunk.title} | "
                    f"пункт {chunk.clause} | "
                    f"score={result.score:.4f}"
                )

    assert len(questions) == 18
