"""Диагностика поиска на демо-вопросах: что ожидалось, что найдено, на каком месте."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from demo_expectations import EXPECTED  # noqa: E402

from app.llm_client import AIClient  # noqa: E402
from app.retrieval.index import KnowledgeBase  # noqa: E402

kb = KnowledgeBase.from_zip(ROOT / "data/documents/knowledge-base.zip", ai=AIClient())
kb.build_vectors()
rows = list(csv.DictReader((ROOT / "data/demo_questions.csv").open(encoding="utf-8-sig")))
lines = ["# Диагностика поиска (демо-вопросы)", "", f"Режим: `{kb.search_mode}`; фрагментов: {len(kb.chunks)}", "",
         "| id | ожидаемое доказательство | место в выдаче | в наборе источников | цели / неоднозначность | топ-3 выдачи |", "|---|---|---|---|---|---|"]
found_top5 = found_evidence = total = 0
for row in rows:
    query = kb.analyze(row["question"])
    hits = kb.search(query)
    evidence = kb.select_evidence(query, hits)
    top = "; ".join(f"{kb.docs[h.chunk.doc_id].short_title} {h.chunk.label}" for h in hits[:3])
    targets = ", ".join(kb.docs[d].number or kb.docs[d].short_title for d in query.targets) or "—"
    if query.ambiguous:
        targets += " / неоднозначно: " + ", ".join(kb.docs[d].number or "" for d in query.ambiguous)
    expected = EXPECTED[row["id"]]
    if not expected:
        lines.append(f"| {row['id']} | (ответа в базе нет) | — | — | {targets}; нет в базе: {', '.join(query.missing_terms) or '—'} | {top} |")
        continue
    for name, clause in expected:
        total += 1
        match = lambda c: name in kb.docs[c.doc_id].filename and (clause is None or c.number == clause)
        rank = next((i for i, h in enumerate(hits, 1) if match(h.chunk)), None)
        in_evidence = any(match(c) for c in evidence)
        found_top5 += bool(rank and rank <= 5)
        found_evidence += in_evidence
        lines.append(f"| {row['id']} | {name} п.{clause} | {rank or 'нет'} | {'да' if in_evidence else 'НЕТ'} | {targets} | {top} |")
lines += ["", f"Ожидаемых доказательств: {total}; в топ-5 выдачи: {found_top5}; в наборе источников для модели: {found_evidence}."]
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports/retrieval_report.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
