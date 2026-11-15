
from pathlib import Path

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents
from app.retrieval.bm25_search import (
    BM25Retriever,
    prepare_query,
    tokenize,
)


def test_q9_diagnostic():

    project_root = Path(__file__).resolve().parents[1]

    archive_path = (
        project_root
        / "data"
        / "documents"
        / "knowledge-base.zip"
    )

    documents = load_documents_from_zip(archive_path)
    chunks = chunk_documents(documents)

    retriever = BM25Retriever(chunks)

    query = (
        "По договору с «Альфа-Трейдинг» "
        "можем оплатить технику авансом "
        "или только после поставки?"
    )

    # Находим именно пункт 4.2 нужного договора.
    target_index = None

    for index, chunk in enumerate(chunks):
        if (
            "ПТ-05-2023" in chunk.filename
            and chunk.clause == "4.2"
        ):
            target_index = index
            break

    assert target_index is not None, (
        "Не найден пункт 4.2 договора ПТ-05-2023"
    )

    target = chunks[target_index]

    query_tokens = prepare_query(query)
    target_tokens = set(tokenize(target.text))

    print("\n" + "=" * 70)
    print("ДИАГНОСТИКА ВОПРОСА №9")
    print("=" * 70)

    print("\nЗапрос:")
    print(query)

    print("\nТокены запроса:")
    print(query_tokens)

    print("\nЦелевой документ:")
    print(target.filename)

    print("\nЦелевой пункт:")
    print(target.clause)

    print("\nТекст пункта:")
    print(target.text)

    # Проверяем вклад каждого токена отдельно.
    print("\nВКЛАД ОТДЕЛЬНЫХ СЛОВ:")

    for token in query_tokens:

        if token not in target_tokens:
            continue

        token_score = retriever.index.get_scores(
            [token]
        )[target_index]

        token_idf = retriever.index.idf.get(
            token,
            0
        )

        print(
            f"{token:20} "
            f"IDF={token_idf:10.4f} "
            f"Score={token_score:10.4f}"
        )

    # Считаем итоговую оценку без фильтрации результатов.
    full_score = retriever.index.get_scores(
        query_tokens
    )[target_index]

    print("\nИТОГОВАЯ ОЦЕНКА ЦЕЛЕВОГО ПУНКТА:")
    print(f"{full_score:.6f}")

    print("\nПОИСКОВАЯ ВЫДАЧА:")

    results = retriever.search(
        query=query,
        top_k=10
    )

    for position, result in enumerate(results, start=1):

        print(
            f"{position}. "
            f"{result.chunk.filename} | "
            f"Пункт {result.chunk.clause} | "
            f"Score: {result.score:.4f}"
        )
