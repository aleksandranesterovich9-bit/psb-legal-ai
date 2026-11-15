from types import SimpleNamespace

from app.retrieval.status_resolver import (
    extract_as_of_date,
    filter_current_chunks,
    is_historical_query,
)


def make_chunk(title, status, valid_from=None, valid_to=None):
    return SimpleNamespace(
        title=title,
        filename=title,
        status=status,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def test_detects_month_and_year_as_historical_date():
    query = "Договор заключали в августе 2024 года. Нужно ли одобрение?"

    assert is_historical_query(query)
    assert extract_as_of_date(query).isoformat() == "2024-08-31"


def test_selects_revision_valid_at_requested_date():
    old_revision = make_chunk(
        "Регламент ред 2023",
        "repealed",
        valid_from="2023-01-10",
        valid_to="2025-03-01",
    )
    current_revision = make_chunk(
        "Регламент ред 2025",
        "active",
        valid_from="2025-03-01",
    )

    results = filter_current_chunks(
        "В августе 2024 года какой порядок действовал?",
        [old_revision, current_revision],
    )

    assert results == [old_revision]


def test_current_query_excludes_repealed_revision():
    old_revision = make_chunk("Регламент ред 2023", "repealed")
    current_revision = make_chunk("Регламент ред 2025", "active")

    results = filter_current_chunks(
        "Кто сейчас инициирует согласование?",
        [old_revision, current_revision],
    )

    assert results == [current_revision]
