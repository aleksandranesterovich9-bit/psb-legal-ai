"""Печатает структуру базы знаний: метаданные документов и фрагменты (для отладки)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.chunker import chunk_document  # noqa: E402
from app.ingestion.document_loader import load_documents_from_zip  # noqa: E402

archive = sys.argv[1] if len(sys.argv) > 1 else "data/documents/knowledge-base.zip"
needle = sys.argv[2] if len(sys.argv) > 2 else None
documents = load_documents_from_zip(archive)
titles = {d.doc_id: d.short_title for d in documents}
for d in documents:
    chunks = chunk_document(d)
    print(f"{d.doc_id} [{d.doc_type}/{d.status}] {d.short_title} | cp={d.counterparties} valid={d.valid_from}..{d.valid_to} "
          f"links={[titles[x] for x in d.links]} injections={len(d.injections)} chunks={len(chunks)}")
    if needle and needle in d.filename:
        for c in chunks:
            print(f"    [{c.kind}|{c.label}|since={c.edition_since}|{len(c.text)}] {c.text[:100]!r}")
