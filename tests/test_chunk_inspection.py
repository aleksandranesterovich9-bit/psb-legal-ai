from app.ingestion.document_loader import load_documents_from_zip
from app.ingestion.chunker import chunk_documents


def test_inspect_required_clauses():
    documents = load_documents_from_zip(
        "data/documents/knowledge-base.zip"
    )

    chunks = chunk_documents(documents)

    targets = {
        "4.2",
        "6.1",
        "6.3",
    }

    for chunk in chunks:
        if chunk.clause in targets:
            print("\n" + "=" * 60)
            print(f"Документ: {chunk.title}")
            print(f"Пункт: {chunk.clause}")
            print(f"Фрагмент: {chunk.text[:700]}")
            