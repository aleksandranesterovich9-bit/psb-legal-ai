import csv

import pytest
from conftest import ROOT
from demo_expectations import EXPECTED

from app.retrieval.index import parse_dialogue

QUESTIONS = {row["id"]: row["question"] for row in csv.DictReader((ROOT / "data/demo_questions.csv").open(encoding="utf-8-sig"))}
CASES = [(qid, name, clause) for qid, items in EXPECTED.items() for name, clause in items]


def evidence_for(kb, question):
    query = kb.analyze(question)
    return query, kb.select_evidence(query, kb.search(query))


def number(kb, ids):
    return [kb.docs[d].number for d in ids]


@pytest.mark.parametrize("qid,name,clause", CASES)
def test_demo_evidence_reaches_model(kb, qid, name, clause):
    _, evidence = evidence_for(kb, QUESTIONS[qid])
    assert any(name in kb.docs[c.doc_id].filename and (clause is None or c.number == clause) for c in evidence)


def test_similar_company_names_are_not_confused(kb):
    assert number(kb, kb.analyze("Какая гарантия по договору с «Альфа-Трейд»?").targets) == ["П-17/2024"]
    assert number(kb, kb.analyze("Какая гарантия по договору с Альфа-Трейдингом?").targets) == ["ПТ-05/2023"]
    _, evidence = evidence_for(kb, "Какой срок оплаты по договору с ООО «Альфа-Трейдинг»?")
    assert not any(kb.docs[c.doc_id].number == "П-17/2024" for c in evidence)


def test_ambiguous_name_and_disambiguation(kb):
    query, evidence = evidence_for(kb, "Подскажите гарантию по договору с «Альфой».")
    assert not query.targets and sorted(number(kb, query.ambiguous)) == ["П-17/2024", "ПТ-05/2023"]
    assert {"П-17/2024", "ПТ-05/2023"} <= {kb.docs[c.doc_id].number for c in evidence}
    assert number(kb, kb.analyze("Мы купили мебель у Альфы. Какая гарантия?").targets) == ["П-17/2024"]


def test_contract_number_variants_and_inflection(kb):
    for text in ("условия п-17-2024", "по договору П17/2024", "договор № П-17/2024"):
        assert number(kb, kb.analyze(text).targets) == ["П-17/2024"]
    assert number(kb, kb.analyze("Что с арендой у Северного Склада?").targets) == ["А-08/2024"]


def test_dialogue_context_carries_contract(kb):
    assert [role for role, _ in parse_dialogue(QUESTIONS["15"])] == ["user", "assistant", "user"]
    raw = "Пользователь: Вопрос по договору П-17/2024.\nАссистент: Что уточнить?\nПользователь: Какой срок ответа на претензию?"
    query, evidence = evidence_for(kb, raw)
    assert number(kb, query.targets) == ["П-17/2024"] and query.question == "Какой срок ответа на претензию?"
    assert any(c.number == "8.1" and kb.docs[c.doc_id].number == "П-17/2024" for c in evidence)


def test_temporal_edition_selection(kb):
    query, evidence = evidence_for(kb, QUESTIONS["5"])
    assert str(query.period[0]) == "2024-08-01" and query.historical
    assert {"repealed", "active"} <= {kb.docs[c.doc_id].status for c in evidence}
    _, current = evidence_for(kb, "Нужно ли одобрение генерального директора для договора на 12 млн рублей?")
    assert not any(kb.docs[c.doc_id].status == "repealed" for c in current)  # без даты — только действующая редакция


def test_amendment_follows_contract(kb):
    _, evidence = evidence_for(kb, "По какому адресу доставляется товар по договору П-17/2024?")
    assert any(kb.docs[c.doc_id].doc_type == "amendment" for c in evidence)


def test_supplementing_letter_follows_regulation(kb):
    _, evidence = evidence_for(kb, "Как согласовать договор с иностранным контрагентом?")
    assert {"regulation", "letter"} <= {kb.docs[c.doc_id].doc_type for c in evidence}


@pytest.mark.parametrize("question,part", [
    ("Какая ставка НДС применяется по общему правилу?", "НК_ст164"),
    ("Какой сейчас минимальный размер оплаты труда?", "МРОТ"),
    ("Сколько часов переработки допускается в год?", "ТК_ст99"),
    ("Как оплачиваются сверхурочные?", "ТК_ст152"),
    ("Какой штраф грозит компании за утечку персональных данных?", "КоАП"),
    ("Какая ставка налога на прибыль?", "НК_ст284"),
    ("Какие обстоятельства смягчают ответственность за налоговое правонарушение?", "НК_ст112"),
])
def test_law_paraphrases(kb, question, part):
    _, evidence = evidence_for(kb, question)
    assert any(part in kb.docs[c.doc_id].filename for c in evidence)


def test_missing_topic_is_flagged(kb):
    assert "пособия" in kb.analyze(QUESTIONS["3"]).missing_terms
    assert not kb.analyze(QUESTIONS["10"]).missing_terms
