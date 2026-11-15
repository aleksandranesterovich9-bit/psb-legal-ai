
from pathlib import Path

from app.ingestion.document_loader import load_documents_from_zip


def test_load_knowledge_base():
    # Определяем путь к архиву относительно корня проекта.
    project_root = Path(__file__).resolve().parents[1]
    archive_path = (
        project_root
        / "data"
        / "documents"
        / "knowledge-base.zip"
    )

    # Загружаем документы.
    documents = load_documents_from_zip(archive_path)

    # Проверяем, что документы действительно загрузились.
    assert len(documents) > 0

    # Проверяем, что у каждого документа есть основные данные.
    for document in documents:
        assert document.document_id
        assert document.filename
        assert document.title
        assert document.text

    # Проверяем уникальность идентификаторов.
    document_ids = [
        document.document_id
        for document in documents
    ]

    assert len(document_ids) == len(set(document_ids))

    # Выводим информацию о загрузке.
    print(f"\nУспешно загружено документов: {len(documents)}")

    for document in documents:
        print(
            f"\nНазвание: {document.title}"
            f"\nКатегория: {document.category}"
            f"\nСтатус: {document.status}"
            f"\nРазмер текста: {len(document.text)} символов"
            f"\nИсточник: {document.source_url}"
        )
