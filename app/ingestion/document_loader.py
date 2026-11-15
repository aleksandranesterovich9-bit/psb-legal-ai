"""Загрузка базы знаний из ZIP и извлечение юридических метаданных документов."""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

from app.generation.security import sanitize_document_text
from app.retrieval import text as tx

SUPPORTED_FORMATS = {".txt", ".md", ".docx"}
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

_ORG_RE = re.compile(r"\b(ООО|АО|ПАО|ЗАО|ИП)\s*[«\"]([^»\"]{2,60})[»\"]")
_LAW_CODES = {
    "ТК": ("ТК РФ", "Трудовой кодекс Российской Федерации"),
    "НК": ("НК РФ", "Налоговый кодекс Российской Федерации"),
    "КОАП": ("КоАП РФ", "Кодекс Российской Федерации об административных правонарушениях"),
    "ГК": ("ГК РФ", "Гражданский кодекс Российской Федерации"),
    "УК": ("УК РФ", "Уголовный кодекс Российской Федерации"),
    "АПК": ("АПК РФ", "Арбитражный процессуальный кодекс Российской Федерации"),
    "ГПК": ("ГПК РФ", "Гражданский процессуальный кодекс Российской Федерации"),
}


@dataclass
class EditionNote:
    """Отметка справочной системы об изменении статьи или её части."""

    scope: str          # «статья», «часть», «пункт», «подпункт», «примечания»
    target: str | None  # номер части/пункта, если указан
    effective: date | None
    text: str


@dataclass
class SourceDocument:
    doc_id: str
    filename: str
    category: str                 # law | company
    doc_type: str                 # contract | amendment | regulation | letter | memo | law | other
    title: str                    # полное наименование
    short_title: str              # краткое наименование для ссылок
    text: str                     # текст после удаления инструкций для ИИ
    raw_text: str
    number: str | None = None
    doc_date: date | None = None
    counterparties: list[str] = field(default_factory=list)
    status: str = "active"        # active | repealed
    valid_from: date | None = None
    valid_to: date | None = None  # первая дата, когда редакция уже не действует
    edition_label: str | None = None
    family: str | None = None     # общий ключ редакций одного документа
    parent_number: str | None = None
    amended_clauses: list[str] = field(default_factory=list)
    referenced_clauses: list[str] = field(default_factory=list)
    subject: str | None = None
    law_code: str | None = None
    law_name: str | None = None
    law_article: str | None = None
    law_article_title: str | None = None
    source_url: str | None = None
    extracted_at: date | None = None
    edition_notes: list[EditionNote] = field(default_factory=list)
    injections: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)   # doc_id связанных документов

    @property
    def is_law(self) -> bool:
        return self.category == "law"

    def effective_on(self, moment: date) -> bool:
        if self.valid_from and moment < self.valid_from:
            return False
        if self.valid_to and moment >= self.valid_to:
            return False
        return True


def extract_docx_text(raw: bytes) -> str:
    """Извлекает текст DOCX без сторонних библиотек, сохраняя абзацы."""

    with zipfile.ZipFile(io.BytesIO(raw)) as docx:
        xml_content = docx.read("word/document.xml")
    root = ET.fromstring(xml_content)
    paragraphs = []
    for paragraph in root.iter(_WORD_NS + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(_WORD_NS + "t"))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _first_date(text: str) -> date | None:
    dates = tx.find_dates(text)
    return dates[0][0] if dates else None


def _organizations(text: str) -> list[str]:
    """Организации документа; написание ПРОПИСНЫМИ приводится к обычному."""

    found: dict[str, str] = {}
    for match in _ORG_RE.finditer(text):
        inner = match.group(2).strip()
        if inner.isupper():
            inner = " ".join("-".join(part.capitalize() for part in word.split("-")) for word in inner.split())
        name = f"{match.group(1)} «{inner}»"
        found.setdefault(tx.normalize(name), name)
    return list(found.values())


def _classify_company_document(first_lines: str) -> str:
    head = tx.normalize(first_lines)
    if "дополнительное соглашение" in head:
        return "amendment"
    if re.search(r"\bдоговор\b", head.split("\n")[0]) or head.startswith("договор"):
        return "contract"
    if "регламент" in head or "положение" in head or "политика" in head or "порядок" in head.split("\n")[0]:
        return "regulation"
    if "служебная записка" in head:
        return "memo"
    if re.search(r"\bисх\.|письмо|уважаемые", head):
        return "letter"
    if "приказ" in head:
        return "order"
    return "other"


def _parse_validity(text: str) -> tuple[date | None, date | None, str | None]:
    """Границы действия редакции внутреннего документа и подпись редакции."""

    head = "\n".join(text.split("\n")[:8])
    normalized = tx.normalize(head)
    valid_from = valid_to = None
    label = None

    match = re.search(r"утратил[аои]?\s+силу\s+с\s+([\d.]+|\d{1,2}\s+[а-я]+\s+\d{4})", normalized)
    if match:
        valid_to = _first_date(match.group(1))
    match = re.search(r"действует\s+с\s+([\d.]+|\d{1,2}\s+[а-я]+\s+\d{4})", normalized)
    if match:
        valid_from = _first_date(match.group(1))
    match = re.search(r"редакция\s+от\s+([\d.]+|\d{1,2}\s+[а-я]+\s+\d{4})", normalized)
    if match:
        edition_date = _first_date(match.group(1))
        if edition_date:
            label = f"редакция от {tx.format_date(edition_date)}"
            if valid_from is None:
                valid_from = edition_date
    return valid_from, valid_to, label


def _family_key(title: str) -> str:
    """Ключ, одинаковый для разных редакций одного документа."""

    key = tx.normalize(title)
    key = re.sub(r"\(.*?\)", " ", key)
    key = re.sub(r"\b(ред|редакция|утратил[аои]? силу)\b\.?", " ", key)
    key = re.sub(r"\b(19|20)\d{2}\b", " ", key)
    return " ".join(tx.tokens(key))


def _company_metadata(document: SourceDocument) -> None:
    lines = [line.strip() for line in document.text.split("\n") if line.strip()]
    head = "\n".join(lines[:6])
    document.doc_type = _classify_company_document(head)

    title_line = next((line for line in lines[:4] if re.search(r"[А-ЯЁ]{4,}", line) and not re.match(r"^(ООО|АО|ПАО)\b", line) and "УТРАТИЛ" not in line.upper()), lines[0])
    organizations = _organizations(document.text)
    # «Своя» организация встречается во всех документах; контрагенты — остальные.
    document.counterparties = organizations

    numbers = tx.find_contract_numbers(head)
    reg_numbers = tx.find_registration_numbers(head)
    document.doc_date = _first_date("\n".join(lines[:5]))

    if document.doc_type == "amendment":
        own = re.search(r"№\s*(\d+)", lines[0])
        document.number = own.group(1) if own else None
        document.parent_number = numbers[0] if numbers else None
        parent_date_text = lines[1] if len(lines) > 1 else ""
        own_date_candidates = tx.find_dates("\n".join(lines[2:5]))
        document.doc_date = own_date_candidates[0][0] if own_date_candidates else _first_date(parent_date_text)
        document.amended_clauses = re.findall(
            r"(?:изложить|дополнить|исключить|изменить)\s+(?:пункт|п\.)\s*(\d+(?:\.\d+)*)", tx.normalize(document.text)
        )
        document.title = f"Дополнительное соглашение № {document.number or '?'} к договору № {document.parent_number or '?'}"
        date_part = f" от {tx.format_date(document.doc_date)}" if document.doc_date else ""
        document.short_title = f"Доп. соглашение № {document.number or '?'}{date_part} к договору № {document.parent_number or '?'}"
        document.valid_from = document.doc_date
    elif document.doc_type == "contract":
        document.number = numbers[0] if numbers else None
        kind = title_line.split("№")[0].strip().capitalize()
        document.title = f"{kind} № {document.number}" if document.number else kind
        short_kind = re.sub(r"\s+(нежилого помещения|недвижимого имущества)$", "", kind, flags=re.I)
        document.short_title = f"{short_kind} № {document.number}" if document.number else short_kind
        subject = next((line for line in lines if re.match(r"^1\.1\.", line)), None)
        document.subject = subject
        document.valid_from = document.doc_date
    else:
        document.number = reg_numbers[0] if reg_numbers else None
        if document.doc_type == "regulation":
            # Номер приказа об утверждении — не номер самого регламента.
            document.number = None
            valid_from, valid_to, label = _parse_validity(document.text)
            document.valid_from, document.valid_to, document.edition_label = valid_from, valid_to, label
            document.doc_date = valid_from or document.doc_date
            name = re.sub(r"\s+", " ", title_line).strip()
            name = name[:1].upper() + name[1:].lower()
            name = re.sub(r"ооо «([^»]+)»", lambda m: "ООО «" + m.group(1).title() + "»", name)
            document.title = name
            document.short_title = f"{name.split(' ООО')[0]} ({label})" if label else name.split(" ООО")[0]
            document.family = _family_key(name)
            if valid_to is not None or re.search(r"утратил[аои]?\s+силу", tx.normalize(lines[0])):
                document.status = "repealed"
        elif document.doc_type == "memo":
            subject = next((line for line in lines if re.match(r"^(О|Об)\s", line)), "")
            date_part = f" от {tx.format_date(document.doc_date)}" if document.doc_date else ""
            document.title = f"Служебная записка № {document.number}{date_part}" + (f" «{subject}»" if subject else "")
            document.short_title = f"Служебная записка № {document.number}{date_part}"
            document.subject = subject or None
            document.valid_from = document.doc_date
        elif document.doc_type == "letter":
            subject = next((line for line in lines if re.match(r"^(О|Об)\s", line)), "")
            date_part = f" от {tx.format_date(document.doc_date)}" if document.doc_date else ""
            document.title = f"Письмо № {document.number}{date_part}" + (f" «{subject}»" if subject else "")
            document.short_title = f"Письмо службы комплаенс № {document.number}{date_part}" if "комплаенс" in tx.normalize(head) else f"Письмо № {document.number}{date_part}"
            document.subject = subject or None
            effective = re.search(r"(?:^|\n|\.\s)С\s+(\d{1,2}\s+[а-яё]+\s+\d{4})", document.text)
            document.valid_from = _first_date(effective.group(1)) if effective else document.doc_date
        else:
            document.title = re.sub(r"\s+", " ", title_line)
            document.short_title = document.title

    # Ссылки на пункты других документов («п. 5.4 Регламента ...»).
    document.referenced_clauses = re.findall(r"\bп\.\s*(\d+(?:\.\d+)+)\s+[А-ЯЁ]", document.text)


def _parse_edition_notes(text: str) -> list[EditionNote]:
    notes: list[EditionNote] = []
    pattern = re.compile(
        r"^(Статья|Часть|Пункт|Подпункт|Абзац|Примечани[ея]|Наименование)\s*([\d.\-]+)?\s*"
        r"(?:стать[ия]\s+[\d.]+\s+)?(изменен[аоы]?|дополнен[аоы]?|изложен[аоы]?)\b(.*)$",
        re.I,
    )
    for line in text.split("\n"):
        line = line.strip()
        match = pattern.match(line)
        if not match:
            continue
        scope = tx.normalize(match.group(1))
        target = match.group(2).rstrip(".") if match.group(2) else None
        # «Статья 9 дополнена частью 9 с ...» — отметка относится к новой части.
        added = re.search(r"дополнен[аоы]?\s+(частью|пунктом|подпунктом)\s+([\d.\-]+)", line, re.I)
        if added:
            scope = {"частью": "часть", "пунктом": "пункт", "подпунктом": "подпункт"}[added.group(1).lower()]
            target = added.group(2).rstrip(".")
        elif scope.startswith("стать"):
            scope, target = "статья", None
        effective_match = re.search(r"\bс\s+(\d{1,2}\s+[а-яё]+\s+\d{4})", line)
        effective = _first_date(effective_match.group(1)) if effective_match else None
        notes.append(EditionNote(scope=scope, target=target, effective=effective, text=line))
    return notes


def _law_metadata(document: SourceDocument) -> None:
    document.doc_type = "law"
    text = document.text
    source_line = re.search(r"^Источник:\s*(.+)$", text, re.M)
    url = re.search(r"^URL:\s*(\S+)", text, re.M)
    extracted = re.search(r"^Дата выгрузки:\s*(.+)$", text, re.M)
    document.source_url = url.group(1) if url else None
    document.extracted_at = _first_date(extracted.group(1)) if extracted else None

    heading = re.search(r"^Статья\s+(\d+(?:\.\d+)*)\.[ \t]+(.*)$", text, re.M)
    if heading:
        document.law_article = heading.group(1)
        title = heading.group(2).strip()
        # Заголовок без точки — название статьи; предложение с точкой — уже норма.
        document.law_article_title = title if title and not title.endswith(".") else None

    source = source_line.group(1) if source_line else ""
    code_short = code_name = None
    stem_name = Path(document.filename).stem.split("__")[-1]
    prefix = re.split(r"[_\s]", stem_name)[0].upper()
    if prefix in _LAW_CODES:
        code_short, code_name = _LAW_CODES[prefix]
    else:
        in_parentheses = re.search(r"\(([А-Яа-яЁё]{1,6}\s+РФ)\)", source)
        federal = re.search(r"Федеральный закон от [\d.]+\s*(?:г\.)?\s*N\s*(\d+-ФЗ)\s*\"([^\"]+)\"", source)
        if in_parentheses:
            code_short = code_name = in_parentheses.group(1)
        elif federal:
            code_short = f"Федеральный закон № {federal.group(1)}"
            code_name = f"Федеральный закон № {federal.group(1)} «{federal.group(2)}»"
    if code_short is None:
        code_short = code_name = stem_name

    document.law_code = code_short
    document.law_name = code_name
    article = f"ст. {document.law_article}" if document.law_article else ""
    document.short_title = f"{code_short}, {article}".rstrip(", ")
    title_suffix = f" «{document.law_article_title}»" if document.law_article_title else ""
    document.title = f"{code_name}, статья {document.law_article}{title_suffix}" if document.law_article else code_name
    if document.law_article and code_name != code_short and "Федеральный закон" in code_short:
        document.short_title = f"{code_name}, ст. {document.law_article}"
    document.edition_notes = _parse_edition_notes(text)
    document.status = "active"


def build_document(filename: str, raw_text: str, index: int) -> SourceDocument:
    """Создаёт документ с метаданными из имени файла и текста."""

    raw_text = raw_text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    sanitized = sanitize_document_text(raw_text)
    prefix = Path(filename).name.split("__", maxsplit=1)[0].lower()
    looks_like_law = bool(re.search(r"^Статья\s+\d", sanitized.text, re.M)) and bool(re.search(r"^Источник:", sanitized.text, re.M))
    category = "law" if prefix.startswith("закон") or (looks_like_law and not prefix.startswith("документ")) else "company"

    digest = hashlib.sha1(f"{filename}\n{raw_text}".encode("utf-8")).hexdigest()[:8]
    document = SourceDocument(
        doc_id=f"d{index:02d}-{digest}",
        filename=filename,
        category=category,
        doc_type="other",
        title=Path(filename).stem,
        short_title=Path(filename).stem,
        text=sanitized.text,
        raw_text=raw_text,
        injections=sanitized.removed,
    )

    if category == "law":
        _law_metadata(document)
    else:
        _company_metadata(document)

    lowered_name = tx.normalize(filename)
    if "утратил" in lowered_name and document.status != "repealed":
        document.status = "repealed"
    return document


def _link_documents(documents: list[SourceDocument]) -> None:
    """Связывает редакции, дополнительные соглашения и дополняющие письма."""

    # «Своя» организация — та, что фигурирует в наибольшем числе документов.
    counts: dict[str, int] = {}
    for document in documents:
        for name in document.counterparties:
            counts[name] = counts.get(name, 0) + 1
    company_docs = [d for d in documents if not d.is_law]
    own = {name for name, count in counts.items() if company_docs and count >= max(3, len(company_docs) // 2 + 1)}
    for document in documents:
        document.counterparties = [name for name in document.counterparties if name not in own]

    by_number = {tx.loose_key(d.number): d for d in documents if d.number and d.doc_type == "contract"}
    for document in documents:
        if document.doc_type == "amendment" and document.parent_number:
            parent = by_number.get(tx.loose_key(document.parent_number))
            if parent:
                document.links.append(parent.doc_id)
                parent.links.append(document.doc_id)
                if not document.counterparties:
                    document.counterparties = list(parent.counterparties)

    # Редакции одного документа: выстраиваем по дате и замыкаем периоды действия.
    families: dict[str, list[SourceDocument]] = {}
    for document in documents:
        if document.family:
            families.setdefault(document.family, []).append(document)
    for members in families.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda d: d.valid_from or date.min)
        for older, newer in zip(members, members[1:]):
            if older.valid_to is None and newer.valid_from:
                older.valid_to = newer.valid_from
            if older.valid_to and older.status != "repealed":
                older.status = "repealed"
            older.links.append(newer.doc_id)
            newer.links.append(older.doc_id)

    # Письма и записки, дополняющие пункт регламента («в дополнение к ... п. 5.4 Регламента»).
    regulations = [d for d in documents if d.doc_type == "regulation"]
    for document in documents:
        if document.doc_type not in {"letter", "memo", "order"}:
            continue
        normalized = tx.normalize(document.text)
        for regulation in regulations:
            family_tokens = (regulation.family or "").split()[:2]
            if family_tokens and all(token in tx.tokens(normalized) for token in family_tokens) and document.referenced_clauses:
                if regulation.edition_label and regulation.edition_label.split()[-1] in document.text:
                    document.links.append(regulation.doc_id)
                    regulation.links.append(document.doc_id)

    for document in documents:
        document.links = list(dict.fromkeys(document.links))


def load_documents_from_zip(archive_path: str | Path) -> list[SourceDocument]:
    """Загружает TXT, MD и DOCX из архива базы знаний.

    Нечитаемые файлы пропускаются с сохранением остальных: один битый
    документ не должен останавливать работу ассистента.
    """

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Архив базы знаний не найден: {archive_path}")

    documents: list[SourceDocument] = []
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for item in sorted(archive.infolist(), key=lambda entry: entry.filename):
                filename = item.filename
                if item.is_dir() or filename.startswith("__MACOSX/") or Path(filename).name.startswith("."):
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_FORMATS:
                    continue
                raw = archive.read(item)
                try:
                    text = extract_docx_text(raw) if suffix == ".docx" else _decode(raw)
                except (zipfile.BadZipFile, KeyError, ET.ParseError, ValueError):
                    continue
                if not text.strip():
                    continue
                documents.append(build_document(filename, text, len(documents) + 1))
    except zipfile.BadZipFile as error:
        raise ValueError(f"Файл не является корректным ZIP-архивом: {archive_path}") from error

    if not documents:
        raise ValueError("В архиве не найдено ни одного поддерживаемого документа.")

    _link_documents(documents)
    return documents
