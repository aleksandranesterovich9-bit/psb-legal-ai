
import hashlib
import re

from dataclasses import dataclass
from app.ingestion.document_loader import SourceDocument


MAX_CHUNK_SIZE = 1800


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    filename: str
    title: str
    category: str
    source_url: str | None
    status: str
    article: str | None
    section: str | None
    part: str | None
    clause: str | None
    subpoint: str | None
    chunk_index: int
    text: str
    valid_from: str | None = None
    valid_to: str | None = None


def build_chunk_id(
    document_id: str,
    chunk_index: int,
    text: str
) -> str:
    """Создаёт стабильный ID фрагмента."""

    content = f"{document_id}:{chunk_index}:{text}"

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def split_long_text(
    text: str,
    max_size: int = MAX_CHUNK_SIZE
) -> list[str]:
    """
    Разбивает длинный текст на части.

    Сначала использует границы предложений,
    затем при необходимости — границы слов.
    """

    if len(text) <= max_size:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+|\n+",
        text
    )

    result = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_size:
            if current:
                result.append(current)
                current = ""

            words = sentence.split()
            word_group = ""

            for word in words:

                # Обрабатываем исключительно длинные слова.
                if len(word) > max_size:
                    if word_group:
                        result.append(word_group)
                        word_group = ""

                    for start in range(0, len(word), max_size):
                        result.append(
                            word[start:start + max_size]
                        )

                    continue

                candidate = (
                    f"{word_group} {word}".strip()
                )

                if len(candidate) <= max_size:
                    word_group = candidate
                else:
                    if word_group:
                        result.append(word_group)

                    word_group = word

            if word_group:
                result.append(word_group)

            continue

        candidate = f"{current} {sentence}".strip()

        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                result.append(current)

            current = sentence

    if current:
        result.append(current)

    return result


def build_document_units(
    document: SourceDocument
) -> list[dict]:
    """
    Разбивает документ на структурные единицы.

    Законы:
    статья -> часть -> подпункт.

    Внутренние документы:
    раздел -> пункт договора.

    Остальные документы:
    текстовые блоки.
    """

    lines = [
        line.strip()
        for line in document.text.splitlines()
    ]

    units = []

    article = None
    section = None
    part = None
    subpoint = None

    current_unit = None

    def flush():
        nonlocal current_unit

        if current_unit is None:
            return

        content = "\n".join(
            current_unit["lines"]
        ).strip()

        if content:
            units.append({
                "article": current_unit["article"],
                "section": current_unit["section"],
                "part": current_unit["part"],
                "clause": current_unit["clause"],
                "subpoint": current_unit["subpoint"],
                "text": content,
            })

        current_unit = None

    def start_unit(
        text: str,
        clause: str | None = None,
        subpoint_number: str | None = None
    ):
        nonlocal current_unit

        current_unit = {
            "article": article,
            "section": section,
            "part": part,
            "clause": clause,
            "subpoint": subpoint_number,
            "lines": [text],
        }

    for line in lines:

        # Пустые строки не разрывают логическую единицу.
        if not line:
            continue

        # Пропускаем технические метаданные.
        if re.match(
            r"^(Источник|URL|Дата выгрузки):",
            line,
            re.IGNORECASE
        ):
            continue

        # Заголовок статьи закона.
        article_match = re.match(
            r"^(Статья\s+\d+[.\d]*\.?.*)$",
            line,
            re.IGNORECASE
        )

        if article_match:
            flush()

            article = article_match.group(1)
            part = None
            subpoint = None

            start_unit(article)
            continue

        if document.category == "law":

            # Часть статьи: 1. Текст...
            part_match = re.match(
                r"^(\d+)\.\s+(.+)$",
                line
            )

            if part_match:
                flush()

                part = part_match.group(1)
                subpoint = None

                start_unit(line)
                continue

            # Подпункт: 1) Текст...
            subpoint_match = re.match(
                r"^(\d+)\)\s*(.+)$",
                line
            )

            if subpoint_match:
                flush()

                subpoint = subpoint_match.group(1)

                start_unit(
                    line,
                    subpoint_number=subpoint
                )
                continue

        elif document.category == "company_document":

            # Раздел договора: 1. Предмет договора
            section_match = re.match(
                r"^(\d+)\.\s+(.+)$",
                line
            )

            if section_match:
                flush()

                section = line
                part = None
                subpoint = None

                start_unit(line)
                continue

            # Пункт договора: 1.1. Текст...
            clause_match = re.match(
                r"^(\d+(?:\.\d+)+)\.\s*(.+)$",
                line
            )

            if clause_match:
                flush()

                clause_number = clause_match.group(1)

                start_unit(
                    line,
                    clause=clause_number
                )
                continue

        # Обычная строка продолжает текущую единицу.
        if current_unit is None:
            start_unit(line)
        else:
            current_unit["lines"].append(line)

    flush()

    return units


def build_chunk_context(
    unit: dict,
    category: str
) -> str:
    """Формирует контекст для юридической ссылки."""

    context_parts = []

    if category == "law":

        article = unit["article"]

        if article:
            context_parts.append(article)

            article_match = re.search(
                r"Статья\s+(\d+(?:\.\d+)*)",
                article,
                re.IGNORECASE
            )

            if article_match and unit["part"]:
                article_number = article_match.group(1)

                context_parts.append(
                    f"Часть {unit['part']} статьи {article_number}"
                )

        if unit["subpoint"]:
            context_parts.append(
                f"Подпункт {unit['subpoint']})"
            )

    elif category == "company_document":

        if unit["section"] and unit["clause"]:
            context_parts.append(
                f"Раздел: {unit['section']}"
            )

    return "\n".join(context_parts)


def chunk_document(
    document: SourceDocument,
    max_size: int = MAX_CHUNK_SIZE
) -> list[DocumentChunk]:
    """Преобразует один документ в список фрагментов."""

    if max_size < 200:
        raise ValueError(
            "Размер фрагмента не должен быть меньше 200 символов."
        )

    units = build_document_units(document)

    chunks = []

    for unit in units:

        context = build_chunk_context(
            unit,
            document.category
        )

        text = unit["text"]

        if context and not text.startswith(context):
            text = f"{context}\n{text}"

        text_parts = split_long_text(
            text,
            max_size=max_size
        )

        for text_part in text_parts:

            chunk_index = len(chunks)

            chunks.append(
                DocumentChunk(
                    chunk_id=build_chunk_id(
                        document.document_id,
                        chunk_index,
                        text_part
                    ),
                    document_id=document.document_id,
                    filename=document.filename,
                    title=document.title,
                    category=document.category,
                    source_url=document.source_url,
                    status=document.status,
                    article=unit["article"],
                    section=unit["section"],
                    part=unit["part"],
                    clause=unit["clause"],
                    subpoint=unit["subpoint"],
                    chunk_index=chunk_index,
                    text=text_part,
                    valid_from=document.valid_from,
                    valid_to=document.valid_to,
                )
            )

    return chunks


def chunk_documents(
    documents: list[SourceDocument],
    max_size: int = MAX_CHUNK_SIZE
) -> list[DocumentChunk]:
    """Разбивает список документов на фрагменты."""

    all_chunks = []

    for document in documents:
        document_chunks = chunk_document(
            document,
            max_size=max_size
        )

        all_chunks.extend(document_chunks)

    return all_chunks
