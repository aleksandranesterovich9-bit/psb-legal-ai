
import hashlib
import re
import zipfile

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    filename: str
    title: str
    category: str
    text: str
    source_url: str | None
    extracted_at: str | None
    status: str
    valid_from: str | None = None
    valid_to: str | None = None


SUPPORTED_FORMATS = {".txt", ".md", ".docx"}

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def extract_docx_text(raw: bytes) -> str:
    """Извлекает текст из DOCX, сохраняя разделение абзацев."""

    import io

    with zipfile.ZipFile(io.BytesIO(raw)) as docx:
        xml_content = docx.read("word/document.xml")

    root = ET.fromstring(xml_content)

    paragraphs = []

    for paragraph in root.iter(WORD_NS + "p"):
        text = "".join(
            node.text or ""
            for node in paragraph.iter(WORD_NS + "t")
        )

        if text.strip():
            paragraphs.append(text.strip())

    return "\n".join(paragraphs)


def extract_metadata(text: str) -> dict:
    """Извлекает метаданные из заголовочной части документа."""

    metadata = {
        "source_url": None,
        "extracted_at": None,
    }

    url_match = re.search(
        r"^URL:\s*(https?://\S+)",
        text,
        re.MULTILINE
    )

    date_match = re.search(
        r"^Дата выгрузки:\s*(.+)$",
        text,
        re.MULTILINE
    )

    if url_match:
        metadata["source_url"] = url_match.group(1).strip()

    if date_match:
        metadata["extracted_at"] = date_match.group(1).strip()

    return metadata


def determine_category(filename: str) -> str:
    """Определяет категорию по имени файла."""

    prefix = filename.split("__", maxsplit=1)[0].lower()

    if prefix == "законы":
        return "law"

    if prefix == "документы компании":
        return "company_document"

    return "unknown"


def determine_status(filename: str, text: str = "") -> str:
    """Определяет статус по имени и явным отметкам внутри документа."""

    normalized_name = filename.lower().replace("ё", "е")
    normalized_text = text.lower().replace("ё", "е")

    if "утратил силу" in normalized_name or re.search(
        r"редакция\s+утратила\s+силу", normalized_text
    ):
        return "repealed"

    if "действующая" in normalized_name or re.search(
        r"действует\s+с\s+\d{2}\.\d{2}\.\d{4}", normalized_text
    ):
        return "active"

    return "unknown"


def extract_validity(text: str) -> tuple[str | None, str | None]:
    """Извлекает границы действия редакции в формате ISO YYYY-MM-DD."""

    normalized = text.lower().replace("ё", "е")

    start_match = re.search(
        r"(?:действует\s+с|редакция\s+от)\s*(\d{2}\.\d{2}\.\d{4})",
        normalized,
    )
    end_match = re.search(
        r"(?:редакция\s+)?утратила\s+силу\s+с\s*(\d{2}\.\d{2}\.\d{4})",
        normalized,
    )

    def to_iso(match):
        if not match:
            return None
        day, month, year = match.group(1).split(".")
        return f"{year}-{month}-{day}"

    return to_iso(start_match), to_iso(end_match)


def build_document_id(filename: str, text: str) -> str:
    """Создаёт стабильный идентификатор документа."""

    content = f"{filename}\n{text}"

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def load_documents_from_zip(
    archive_path: str | Path
) -> list[SourceDocument]:
    """
    Загружает документы из ZIP-архива базы знаний.

    Поддерживаемые форматы:
    TXT, MD, DOCX.
    """

    archive_path = Path(archive_path)

    if not archive_path.exists():
        raise FileNotFoundError(
            f"Архив базы знаний не найден: {archive_path}"
        )

    if not archive_path.is_file():
        raise ValueError(
            f"Указанный путь не является файлом: {archive_path}"
        )

    documents = []

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:

            for item in sorted(archive.infolist(), key=lambda x: x.filename):

                if item.is_dir():
                    continue

                filename = item.filename

                if filename.startswith("__MACOSX/"):
                    continue

                suffix = Path(filename).suffix.lower()

                if suffix not in SUPPORTED_FORMATS:
                    continue

                raw = archive.read(item)

                try:
                    if suffix == ".docx":
                        text = extract_docx_text(raw)
                    else:
                        text = raw.decode("utf-8-sig")

                except (
                    UnicodeDecodeError,
                    zipfile.BadZipFile,
                    KeyError,
                    ET.ParseError,
                ) as error:
                    raise ValueError(
                        f"Не удалось прочитать документ {filename}: {error}"
                    ) from error

                text = text.replace("\x00", "").strip()

                if not text:
                    continue

                metadata = extract_metadata(text)

                title = Path(filename).stem

                if "__" in title:
                    title = title.split("__", maxsplit=1)[1]

                valid_from, valid_to = extract_validity(text)

                document = SourceDocument(
                    document_id=build_document_id(filename, text),
                    filename=filename,
                    title=title,
                    category=determine_category(filename),
                    text=text,
                    source_url=metadata["source_url"],
                    extracted_at=metadata["extracted_at"],
                    status=determine_status(filename, text),
                    valid_from=valid_from,
                    valid_to=valid_to,
                )

                documents.append(document)

    except zipfile.BadZipFile as error:
        raise ValueError(
            f"Файл не является корректным ZIP-архивом: {archive_path}"
        ) from error

    if not documents:
        raise ValueError(
            "В архиве не найдено ни одного поддерживаемого документа."
        )

    return documents
