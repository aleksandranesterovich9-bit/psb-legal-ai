import calendar
import re
from datetime import date


HISTORICAL_QUERY_PATTERNS = (
    r"\bв\s+20\d{2}\s+году\b",
    r"\bза\s+20\d{2}\s+год",
    r"\bпо состоянию на\b",
    r"\bна\s+тот момент\b",
    r"\bранее\b",
    r"\bисторическ\w*",
    r"\bсравни\b",
    r"\bсравнение\b",
    r"\bредакци\w*",
    # Даты вида «в августе 2024 года» и «за март 2023 года».
    r"\bв\s+(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма\w*|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+20\d{2}\b",
    r"\bза\s+(?:январ\w*|феврал\w*|март\w*|апрел\w*|ма\w*|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+20\d{2}\b",
)

MONTHS = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "ма": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}


def is_historical_query(query: str) -> bool:
    """Определяет, относится ли вопрос к прошлой редакции документа."""

    normalized = query.lower().replace("ё", "е")
    return any(re.search(pattern, normalized) for pattern in HISTORICAL_QUERY_PATTERNS)


def extract_as_of_date(query: str) -> date | None:
    """Извлекает дату, на которую нужно восстановить состояние документов."""

    normalized = query.lower().replace("ё", "е")

    numeric_date = re.search(r"\b(\d{1,2})[./](\d{1,2})[./](20\d{2})\b", normalized)
    if numeric_date:
        day, month, year = map(int, numeric_date.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    month_pattern = (
        r"\b(?:в|за|на)\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма\w*|"
        r"июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)"
        r"\s+(20\d{2})\b"
    )
    month_match = re.search(month_pattern, normalized)
    if month_match:
        month_word, year_text = month_match.groups()
        month = next(
            number
            for prefix, number in MONTHS.items()
            if month_word.startswith(prefix)
        )
        year = int(year_text)
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, last_day)

    year_match = re.search(r"\b(?:в|за|на)\s+(20\d{2})\s*(?:году|год|г\.)?\b", normalized)
    if year_match:
        return date(int(year_match.group(1)), 12, 31)

    return None


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def filter_current_chunks(query: str, chunks: list) -> list:
    """
    Фильтрует документы по статусу и, если указана дата,
    по периоду действия редакции.

    Для исторического запроса сохраняются только редакции,
    действовавшие на указанную дату. Если дата не указана,
    возвращаются доступные редакции, включая утратившие силу.
    """

    historical = is_historical_query(query)
    as_of = extract_as_of_date(query) if historical else None

    if not historical:
        return [chunk for chunk in chunks if chunk.status != "repealed"]

    if as_of is None:
        return chunks

    filtered = []
    for chunk in chunks:
        valid_from = _parse_iso_date(getattr(chunk, "valid_from", None))
        valid_to = _parse_iso_date(getattr(chunk, "valid_to", None))

        if valid_from and as_of < valid_from:
            continue
        if valid_to and as_of >= valid_to:
            continue

        # Утративший силу документ без даты окончания действия
        # нельзя достоверно привязать к прошлому периоду.
        if chunk.status == "repealed" and valid_to is None:
            continue

        # Резервная проверка для документов с годом редакции в названии.
        edition_match = re.search(
            r"ред(?:акция)?\s*(20\d{2})",
            f"{chunk.title} {chunk.filename}",
            re.IGNORECASE,
        )
        if edition_match and int(edition_match.group(1)) > as_of.year:
            continue

        filtered.append(chunk)

    return filtered


def get_status_multiplier(query: str, chunk) -> float:
    """Немного повышает приоритет действующих документов в обычных запросах."""

    if is_historical_query(query):
        return 1.0

    if chunk.status == "active":
        return 1.25

    return 1.0
