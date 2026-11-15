
from pathlib import Path

from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents


def test_chunk_knowledge_base():
    project_root = Path(__file__).resolve().parents[1]

    archive_path = (
        project_root
        / "data"
        / "documents"
        / "knowledge-base.zip"
    )

    documents = load_documents_from_zip(archive_path)
    chunks = chunk_documents(documents)

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id
        assert chunk.text.strip()
        assert chunk.title

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))

    document_ids = {
        document.document_id
        for document in documents
    }

    for chunk in chunks:
        assert chunk.document_id in document_ids

    # Общая статистика.
    print(f"\nДокументов: {len(documents)}")
    print(f"Фрагментов: {len(chunks)}")

    sizes = [len(chunk.text) for chunk in chunks]

    print(f"Минимальный размер: {min(sizes)}")
    print(f"Максимальный размер: {max(sizes)}")
    print(
        "Фрагментов больше 1800 символов:",
        sum(size > 1800 for size in sizes)
    )

    # Проверяем договор аренды.
    lease_chunks = [
        chunk
        for chunk in chunks
        if "Договор аренды" in chunk.title
    ]

    print("\n" + "=" * 70)
    print("ПОСЛЕДНИЕ ФРАГМЕНТЫ ДОГОВОРА АРЕНДЫ")
    print("=" * 70)

    for chunk in lease_chunks[-10:]:
        print("\n" + "-" * 60)
        print(f"Раздел: {chunk.section}")
        print(f"Пункт: {chunk.clause}")
        print(f"Размер: {len(chunk.text)}")
        print(chunk.text[:700])

    # Проверяем закон 152-ФЗ.
    law_chunks = [
        chunk
        for chunk in chunks
        if "152ФЗ" in chunk.title
    ]

    print("\n" + "=" * 70)
    print("ФРАГМЕНТЫ ЗАКОНА 152-ФЗ")
    print("=" * 70)

    for chunk in law_chunks[:15]:
        print("\n" + "-" * 60)
        print(f"Статья: {chunk.article}")
        print(f"Часть: {chunk.part}")
        print(f"Пункт: {chunk.clause}")
        print(f"Размер: {len(chunk.text)}")
        print(chunk.text[:500])


def test_chunk_size_limit():
    project_root = Path(__file__).resolve().parents[1]

    archive_path = (
        project_root
        / "data"
        / "documents"
        / "knowledge-base.zip"
    )

    documents = load_documents_from_zip(archive_path)
    chunks = chunk_documents(documents)

    oversized = [
        chunk
        for chunk in chunks
        if len(chunk.text) > 1800
    ]

    assert not oversized, (
        f"Найдено фрагментов больше 1800 символов: "
        f"{len(oversized)}"
    )
