"""Точка входа автоматической проверки ПСБ: data/questions.csv -> submission.csv.

Зависимости: стандартная библиотека и httpx. Контракт ввода/вывода и имена
переменных окружения совпадают с официальным примером сабмита.
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:  # в ZIP платформы модули лежат в корне (psb_*.py), в репозитории — пакет app
    from app.generation.engine import LegalAssistant, plain_text  # noqa: E402
    from app.llm_client import AIClient, env, env_int  # noqa: E402
    from app.retrieval.index import KnowledgeBase  # noqa: E402
except ModuleNotFoundError:
    from psb_engine import LegalAssistant, plain_text  # type: ignore[no-redef]  # noqa: E402
    from psb_llm_client import AIClient, env, env_int  # type: ignore[no-redef]  # noqa: E402
    from psb_index import KnowledgeBase  # type: ignore[no-redef]  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logging.getLogger("httpx").setLevel(logging.WARNING)  # не засорять лог платформы строкой на каждый запрос
logger = logging.getLogger("solution")

DATA_DIR = Path(env("DATA_DIR", "data"))
MAX_ANSWER_CHARS = 9500  # лимит платформы — 10 000 символов
FALLBACK = "Не удалось сформировать ответ по базе знаний из-за технической ошибки. Вывод без подтверждения документами не делаю."


def find_file(*names: str) -> Path:
    roots = (DATA_DIR, ROOT / DATA_DIR, Path.cwd(), ROOT, ROOT / "data" / "documents", ROOT / "data")
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return path
    for root in roots:  # запасной вариант: единственный zip в каталоге данных
        archives = sorted(root.glob("*.zip")) if root.is_dir() else []
        if len(archives) == 1 and "kb.zip" in names:
            return archives[0]
    raise FileNotFoundError(f"Не найден ни один из файлов: {', '.join(names)}")


def load_questions(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows = [(str(row["id"]).strip(), str(row["question"] or "").strip()) for row in reader]
    if not rows or any(not question_id for question_id, _ in rows):
        raise ValueError("В questions.csv отсутствуют вопросы или id")
    return rows


def write_submission(rows: list[tuple[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows(rows)


def main() -> None:
    started = time.perf_counter()
    questions = load_questions(find_file("questions.csv", "demo_questions.csv"))
    kb_path = find_file("kb.zip", "knowledge-base.zip", "knowledge_base.zip")

    ai = AIClient()
    ai.deadline = time.monotonic() + env_int("TIME_BUDGET_SEC", 960, 60, 3600)
    kb = KnowledgeBase.from_zip(kb_path, ai=ai)
    kb.build_vectors()
    assistant = LegalAssistant(kb, ai)
    workers = env_int("MAX_WORKERS", 2, 1, 8)
    print(f"Вопросы: {len(questions)}; документы: {len(kb.documents)}; фрагменты: {len(kb.chunks)}; "
          f"поиск: {kb.search_mode}; LLM: {'да' if ai.has_llm else 'нет'}; потоков: {workers}", flush=True)

    def work(position: int, question: str) -> tuple[int, str, str]:
        # Каждый тест-кейс обрабатывается независимо: состояние между вопросами не хранится.
        try:
            result = assistant.answer(question)
            answer = plain_text(result.answer)
            if len(answer) > MAX_ANSWER_CHARS:
                answer = answer[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0] + "…"
            return position, answer or FALLBACK, f"{result.mode}/{result.status}"
        except Exception as error:  # noqa: BLE001 — один вопрос не должен ронять сабмит
            logger.error("Вопрос %s: %s: %s", position, type(error).__name__, error)
            return position, FALLBACK, "error"

    answers = [FALLBACK] * len(questions)
    outcomes: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, index, question) for index, (_, question) in enumerate(questions)]
        for finished, future in enumerate(as_completed(futures), start=1):
            index, answer, outcome = future.result()
            answers[index] = answer
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if finished % 10 == 0 or finished == len(questions):
                print(f"Обработано {finished}/{len(questions)}", flush=True)

    output = Path(os.getenv("OUTPUT_PATH", "submission.csv"))
    write_submission([(question_id, answers[index]) for index, (question_id, _) in enumerate(questions)], output)
    # Сводка для лога платформы: сколько ответов дала модель, а сколько — запасной режим, и почему.
    print("Режимы/статусы ответов: " + ", ".join(f"{key}={value}" for key, value in sorted(outcomes.items())), flush=True)
    if ai.errors or ai.disabled:
        print(f"Ошибки API: {dict(ai.errors)}; отключены: {sorted(ai.disabled)}", flush=True)
    ai.close()
    print(f"Создан {output}: {len(questions)} ответов за {time.perf_counter() - started:.1f} с", flush=True)


if __name__ == "__main__":
    main()
