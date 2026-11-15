"""Структурное разбиение документов на фрагменты.

* Договоры и регламенты: один фрагмент = один пункт (с названием раздела).
* Письма, записки, доп. соглашения: абзацы, короткие строки объединяются.
* Законы: часть/пункт статьи; длинные пункты делятся по подпунктам, при этом
  каждый фрагмент сохраняет вводную фразу пункта, чтобы исключение или
  перечень не отрывались от управляющей нормы. Отметки справочной системы
  («Пункт 3 изменен с ...») не считаются нормой и становятся метаданными.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from app.ingestion.document_loader import SourceDocument
from app.retrieval import text as tx

MAX_CHARS = 1200          # целевой размер фрагмента закона
HARD_MAX_CHARS = 2400     # абзац длиннее делится по предложениям
SHORT_LINE = 80           # короткие строки писем объединяются со следующими

_SECTION_RE = re.compile(r"^(\d{1,2})\.\s+(\S.*)$")
_CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\.?\s+(\S.*)$")
_LAW_UNIT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\.\s+(\S.*)$")
_LAW_SUBITEM_RE = re.compile(r"^(\d+(?:\.\d+)*(?:-\d+)?)\)\s*(\S.*)$")
_ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.\s+(.*)$")
_META_RE = re.compile(r"^(Источник|URL|Дата выгрузки):", re.I)

_EDITORIAL_PATTERNS = [
    re.compile(r"^(Статья|Часть|Пункт|Подпункт|Абзац|Примечани[ея]|Наименование)\b.{0,60}\b(изменен[аоы]?|дополнен[аоы]?|изложен[аоы]?)\b.*\bс\s+\d", re.I),
    re.compile(r"^Федеральн(ым|ыми)\s+закон(ом|ами)\b", re.I),
    re.compile(r"^(О|Об)\s+(проверке конституционности|конституционно-правовом|применении|порядке применения)", re.I),
    re.compile(r"^См\.", re.I),
    re.compile(r"^[CС]рок давности привлечения", re.I),
    re.compile(r"^Обратите внимание", re.I),
    re.compile(r"^Положения\s+(стать|пункт|част|подпункт|абзац)\w*\s.{0,60}\(в\s+ред", re.I),
    re.compile(r"^(Информация об изменениях|ГАРАНТ:)", re.I),
]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    kind: str                     # preamble | clause | section | paragraph | part | note
    label: str                    # «п. 3.1», «ч. 5», «п. 1, пп. 2.1–2.3»
    number: str | None            # номер пункта договора или части/пункта статьи
    section: str | None
    text: str
    search_text: str = ""
    lead: str | None = None       # вводная фраза пункта для продолжений
    subitems: list[str] = field(default_factory=list)
    edition_since: date | None = None
    edition_notes: list[str] = field(default_factory=list)


def is_editorial(line: str) -> bool:
    """Отметка справочной системы, а не текст нормы."""

    return any(pattern.search(line) for pattern in _EDITORIAL_PATTERNS)


def _split_sentences(text: str, limit: int) -> list[str]:
    """Делит сверхдлинный абзац по границам предложений или перечислений."""

    pieces = re.split(r"(?<=[.;])\s+", text)
    result: list[str] = []
    current = ""
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            head, piece = piece[:cut], piece[cut:].lstrip()
            if current:
                result.append(current)
                current = ""
            result.append(head)
        candidate = f"{current} {piece}".strip()
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                result.append(current)
            current = piece
    if current:
        result.append(current)
    return result


# --------------------------------------------------------------------------
# Внутренние документы
# --------------------------------------------------------------------------

def _company_header(document: SourceDocument) -> str:
    parties = ", ".join(document.counterparties)
    return f"{document.short_title}" + (f" ({parties})" if parties and parties not in document.short_title else "")


def _chunk_structured_company(document: SourceDocument, lines: list[str]) -> list[dict]:
    units: list[dict] = []
    section: str | None = None
    current: dict | None = None
    preamble: list[str] = []

    def flush() -> None:
        nonlocal current
        if current is not None:
            units.append(current)
            current = None

    for line in lines:
        clause = _CLAUSE_RE.match(line)
        heading = _SECTION_RE.match(line) if not clause else None
        if clause:
            flush()
            current = {"kind": "clause", "number": clause.group(1), "label": f"п. {clause.group(1)}", "section": section, "lines": [line]}
        elif heading and len(line) <= 90 and not line.rstrip().endswith((".", ";", ":")):
            flush()
            section = line
            current = {"kind": "section", "number": heading.group(1), "label": f"раздел {heading.group(1)}", "section": section, "lines": [line]}
        elif current is None:
            preamble.append(line)
        else:
            current["lines"].append(line)
    flush()

    result: list[dict] = []
    if preamble:
        result.append({"kind": "preamble", "number": None, "label": "преамбула", "section": None, "lines": preamble})
    for unit in units:
        # Заголовок раздела без собственного текста нужен только как контекст пунктов.
        if unit["kind"] == "section" and len(unit["lines"]) == 1:
            continue
        result.append(unit)
    return result


def _chunk_plain_company(lines: list[str]) -> list[dict]:
    units: list[dict] = []
    buffer: list[str] = []

    def flush(kind: str = "paragraph") -> None:
        if buffer:
            units.append({"kind": kind, "number": None, "label": "", "section": None, "lines": list(buffer)})
            buffer.clear()

    for line in lines:
        is_long = len(line) >= SHORT_LINE
        if is_long and buffer and not buffer[-1].rstrip().endswith(":") and sum(map(len, buffer)) >= 40:
            flush()
        buffer.append(line)
        if is_long and not line.rstrip().endswith(":"):
            flush()
    flush()
    if units:
        units[0]["kind"] = "preamble"
        units[0]["label"] = "вводная часть"
    return units


def _chunk_company(document: SourceDocument) -> list[Chunk]:
    lines = [line.strip() for line in document.text.split("\n") if line.strip()]
    clause_count = sum(1 for line in lines if _CLAUSE_RE.match(line) and not line.startswith(("«", '"')))
    units = _chunk_structured_company(document, lines) if clause_count >= 3 else _chunk_plain_company(lines)

    header = _company_header(document)
    chunks: list[Chunk] = []
    for unit in units:
        text = "\n".join(unit["lines"])
        section = unit["section"]
        search_parts = [header]
        if section and unit["kind"] == "clause":
            search_parts.append(section)
        search_parts.append(text)
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{len(chunks)}",
                doc_id=document.doc_id,
                index=len(chunks),
                kind=unit["kind"],
                label=unit["label"],
                number=unit["number"],
                section=section,
                text=text,
                search_text="\n".join(search_parts),
            )
        )
    return chunks


# --------------------------------------------------------------------------
# Законы
# --------------------------------------------------------------------------

def _note_date(note: str) -> date | None:
    match = re.search(r"\bс\s+(\d{1,2}\s+[а-яё]+\s+\d{4})", note)
    if not match:
        return None
    dates = tx.find_dates(match.group(1))
    return dates[0][0] if dates else None


def _chunk_law(document: SourceDocument) -> list[Chunk]:
    raw_lines = [line.strip() for line in document.text.split("\n") if line.strip()]
    lines = [line for line in raw_lines if not _META_RE.match(line)]

    is_tax_code = "НК" in (document.law_code or "")
    unit_word, sub_word = ("п.", "пп.") if is_tax_code else ("ч.", "п.")

    body_started = False
    numbered_mode = False
    for line in lines:
        if _ARTICLE_RE.match(line):
            body_started = True
            continue
        if body_started and not is_editorial(line) and _LAW_UNIT_RE.match(line):
            numbered_mode = True
            break

    header = document.short_title + (f" «{document.law_article_title}»" if document.law_article_title else "")
    chunks: list[Chunk] = []
    pending_notes: list[str] = []
    state = {"unit": None, "lead": None, "word": unit_word, "part_counter": 0, "notes_mode": False, "last_sub": None}
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is None or not current["lines"]:
            current = None
            return
        text = "\n".join(current["lines"])
        number = current["unit"]
        word = current["word"]
        label = f"{word} {number}" if number else "текст статьи"
        subitems = list(current["subitems"])
        if current["continued"] and current["sub_before"] and not _LAW_SUBITEM_RE.match(current["lines"][0]):
            subitems.insert(0, current["sub_before"])
        if current["continued"]:
            if subitems:
                span = subitems[0] if subitems[0] == subitems[-1] else f"{subitems[0]}–{subitems[-1]}"
                label += f", {sub_word} {span}"
            else:
                label += " (продолжение)"
        lead = current["lead"] if current["continued"] else None
        search_parts = [header]
        if lead:
            search_parts.append(lead[:300])
        search_parts.append(text)
        notes = current["notes"]
        dates = [value for value in (_note_date(note) for note in notes) if value]
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{len(chunks)}",
                doc_id=document.doc_id,
                index=len(chunks),
                kind="note" if current["notes_mode"] else "part",
                label=label,
                number=number,
                section=None,
                text=text,
                search_text="\n".join(search_parts),
                lead=lead,
                subitems=list(subitems),
                edition_since=max(dates) if dates else None,
                edition_notes=list(notes),
            )
        )
        current = None

    def start(line: str, continued: bool) -> None:
        nonlocal current
        flush()
        current = {
            "unit": state["unit"], "word": state["word"], "lead": state["lead"],
            "continued": continued, "lines": [], "subitems": [], "notes": [],
            "notes_mode": state["notes_mode"], "sub_before": state["last_sub"],
        }
        add(line)

    def add(line: str) -> None:
        assert current is not None
        if pending_notes:
            current["notes"].extend(pending_notes)
            pending_notes.clear()
        sub = _LAW_SUBITEM_RE.match(line)
        if sub:
            current["subitems"].append(sub.group(1))
            state["last_sub"] = sub.group(1)
        current["lines"].append(line)

    def current_size() -> int:
        return sum(len(item) + 1 for item in current["lines"]) if current else 0

    body_started = False
    for line in lines:
        article = _ARTICLE_RE.match(line)
        if article and not body_started:
            body_started = True
            pending_notes.clear()  # отметки до заголовка относятся ко всей статье
            rest = article.group(2).strip()
            if rest.endswith(".") and len(rest.split()) > 6:
                # «Статья 1. Установить ...» — заголовок уже содержит норму (часть 1).
                state["part_counter"] = 1
                state["unit"], state["lead"] = "1", line
                start(line, continued=False)
            continue
        if not body_started:
            continue
        if is_editorial(line):
            scope_article = re.match(r"^Статья\b", line) and "дополнена" not in line
            if not scope_article:
                pending_notes.append(line)
            continue
        if re.match(r"^Примечани[ея]:?$", line):
            flush()
            state.update({"notes_mode": True, "word": "примечание", "unit": None, "lead": None, "last_sub": None})
            continue

        unit = _LAW_UNIT_RE.match(line) if numbered_mode else None
        starts_part = (not numbered_mode) and line[:1].isupper() and not _LAW_SUBITEM_RE.match(line)

        pieces = [line] if len(line) <= HARD_MAX_CHARS else _split_sentences(line, MAX_CHARS)

        if unit:
            state["last_sub"] = None
            state["unit"], state["lead"] = unit.group(1), pieces[0]
            start(pieces[0], continued=False)
            for extra in pieces[1:]:
                start(extra, continued=True)
        elif starts_part:
            state["part_counter"] += 1
            state["last_sub"] = None
            state["unit"], state["lead"] = str(state["part_counter"]), pieces[0]
            start(pieces[0], continued=False)
            for extra in pieces[1:]:
                start(extra, continued=True)
        else:
            for piece in pieces:
                if current is None:
                    start(piece, continued=state["unit"] is not None)
                elif current_size() + len(piece) > MAX_CHARS and len(current["lines"]) > (0 if current["continued"] else 1):
                    start(piece, continued=True)
                else:
                    add(piece)
    flush()
    return chunks


def chunk_document(document: SourceDocument) -> list[Chunk]:
    return _chunk_law(document) if document.is_law else _chunk_company(document)


def chunk_documents(documents: list[SourceDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks
