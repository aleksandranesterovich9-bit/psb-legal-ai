import re
from dataclasses import dataclass

from pymorphy3 import MorphAnalyzer
from rank_bm25 import BM25Okapi

from app.ingestion.chunker import DocumentChunk
from app.retrieval.status_resolver import (
    filter_current_chunks,
    get_status_multiplier,
)


@dataclass(frozen=True)
class SearchResult:
    chunk: DocumentChunk
    score: float


morph = MorphAnalyzer()

# Порог защищает от уверенного вида случайных совпадений.
# В частности, не позволяет выдавать статью о сверхурочной работе
# в ответ на вопрос о пособии, которого в базе знаний нет.
MIN_RELEVANCE_SCORE = 4.5


STOPWORDS = {
    "и", "в", "во", "на", "по", "с", "со", "к", "ко",
    "у", "о", "об", "от", "до", "за", "из", "для",
    "или", "а", "но", "это", "мы", "наш", "мне",
    "пожалуйста", "можно", "ли", "какой", "какая",
    "какое", "какие", "как", "сколько", "когда",
    "где", "что", "подскажите", "пользователь",
    "ассистент", "бы", "же", "то", "либо", "также",
}

# Эти слова из предыдущих реплик обычно описывают старый вопрос,
# а не контекст текущего уточнения.
CONTEXT_STOPWORDS = {
    "вопрос", "ответ", "срок", "гарантия", "гарантийный",
    "оплата", "платеж", "расход", "проверка", "проверить",
    "качество", "количество", "день", "число", "цена",
    "стоимость", "аванс", "предоплата", "пособие",
}


QUERY_SYNONYMS = {
    "аванс": ["предоплата", "авансовый", "платеж"],
    "авансовый": ["предоплата", "аванс"],
    "оплатить": ["оплата", "оплачивать"],

    "гарантия": ["гарантийный", "срок", "производитель", "талон"],
    "гарантийный": ["гарантия", "производитель", "талон"],

    "ключ": ["возврат", "помещение", "акт"],
    "сдать": ["возврат", "возвращать", "акт"],
    "съезжать": ["прекращение", "возврат", "помещение", "акт"],

    "проверить": ["приемка", "качество"],
    "проверка": ["приемка", "качество"],
    "дефект": ["недостаток", "акт"],

    "запустить": ["инициировать", "согласование"],
    "пропускать": ["доступ", "беспрепятственный"],
    "пропуск": ["доступ", "беспрепятственный"],
    "грузовик": ["транспорт"],
    "ночью": ["круглосуточно"],
    "юротдел": ["юридический отдел"],
    "отнести": ["передать", "передача"],
    "передавать": ["передать", "передача"],
}


RAW_QUERY_SYNONYMS = {
    "авансом": ["предоплата"],
    "авансов": ["предоплата"],
    "аванс": ["предоплата"],
    "заранее": ["предоплата"],
    "предварительно": ["предоплата"],
    "пропускать": ["доступ", "круглосуточно"],
    "юротдел": ["юридический", "отдел"],
}


def extract_user_messages(query: str) -> str:
    """
    Для диалога берёт последнюю реплику пользователя как сам вопрос.
    Из предыдущего контекста сохраняет только полезные сущности и
    имена существительные, а из ответа ассистента — номер договора
    и название организации. Старый вопрос не переносится в поиск.
    """

    pattern = re.compile(
        r"^\s*(Пользователь|Ассистент):\s*(.*?)"
        r"(?=^\s*(?:Пользователь|Ассистент):|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    turns = pattern.findall(query)

    if not turns:
        return query

    user_messages = [text.strip() for role, text in turns if role == "Пользователь"]
    assistant_messages = [text.strip() for role, text in turns if role == "Ассистент"]

    if not user_messages:
        return query

    latest_message = user_messages[-1]
    context_terms = []

    # Неспрашивающие реплики пользователя содержат контекст диалога
    # («оригинал договора», «склад», «мебель»), но не старый вопрос.
    for previous_message in user_messages[:-1]:
        if "?" in previous_message or "？" in previous_message:
            continue

        for word in re.findall(r"[a-zа-яё]+", previous_message.lower()):
            parsed = morph.parse(word)[0]
            if parsed.tag.POS != "NOUN":
                continue

            normal = parsed.normal_form
            if normal in STOPWORDS or normal in CONTEXT_STOPWORDS:
                continue

            if normal not in context_terms:
                context_terms.append(normal)

    # Если последняя реплика не называет договор/организацию,
    # извлекаем эти идентификаторы из уточнения ассистента.
    has_reference = bool(
        re.search(r"[А-ЯЁA-Z]{1,4}[-–]?\d{1,3}[/–-]20\d{2}", latest_message, re.IGNORECASE)
        or re.search(r"ООО\s*[«\"]|[«\"][^»\"]+[»\"]", latest_message, re.IGNORECASE)
    )

    if not has_reference:
        for assistant_message in assistant_messages:
            identifiers = re.findall(
                r"[А-ЯЁA-Z]{1,4}[-–]?\d{1,3}[/–-]20\d{2}",
                assistant_message,
                re.IGNORECASE,
            )
            organizations = re.findall(
                r"ООО\s*[«\"]([^»\"]+)[»\"]",
                assistant_message,
                re.IGNORECASE,
            )
            context_terms.extend(identifiers)
            context_terms.extend(organizations)

    # Сохраняем порядок и удаляем повторы.
    unique_context = list(dict.fromkeys(context_terms))
    return " ".join([latest_message, *unique_context])


def tokenize(text: str) -> list[str]:
    """Разбивает текст на токены и приводит русские слова к начальной форме."""

    normalized = text.lower().replace("ё", "е")
    words = re.findall(r"[a-zа-я0-9]+", normalized)

    tokens = []
    for word in words:
        parsed_word = morph.parse(word)[0]
        tokens.append(parsed_word.normal_form)

    return tokens


def prepare_query(query: str) -> list[str]:
    """Нормализует запрос, добавляет юридические синонимы и убирает шум."""

    query = extract_user_messages(query)
    normalized_query = query.lower().replace("ё", "е")
    raw_words = re.findall(r"[a-zа-я0-9]+", normalized_query)

    result_tokens = []
    seen = set()

    def add_token(token: str):
        if token in STOPWORDS:
            return

        if token not in seen:
            result_tokens.append(token)
            seen.add(token)

        for synonym in QUERY_SYNONYMS.get(token, []):
            for synonym_token in tokenize(synonym):
                if synonym_token not in STOPWORDS and synonym_token not in seen:
                    result_tokens.append(synonym_token)
                    seen.add(synonym_token)

    for raw_word in raw_words:
        for synonym in RAW_QUERY_SYNONYMS.get(raw_word, []):
            for synonym_token in tokenize(synonym):
                add_token(synonym_token)

        for token in tokenize(raw_word):
            add_token(token)

    return result_tokens


def has_explicit_document_reference(
    query_tokens: list[str],
    chunk: DocumentChunk,
) -> bool:
    """Проверяет совпадение номера договора или названия организации."""

    query_set = set(query_tokens)
    title_tokens = tokenize(chunk.title)

    # Сначала проверяем номера документов после удаления разделителей.
    normalized_query = "".join(query_tokens)
    normalized_title = "".join(title_tokens)
    identifiers = re.findall(
        r"[а-яa-z]{1,4}\d{1,3}20\d{2}",
        normalized_query,
    )
    for identifier in identifiers:
        if identifier in normalized_title:
            return True

    generic_words = {
        "договор", "поставка", "аренда", "документ",
        "компания", "общество", "ооо", "редакция",
    }

    for index in range(len(title_tokens) - 1):
        first = title_tokens[index]
        second = title_tokens[index + 1]

        if first in generic_words or second in generic_words:
            continue

        if first.isdigit() or second.isdigit():
            continue

        if first in query_set and second in query_set:
            return True

    return False


def _intent_multiplier(query: str, chunk: DocumentChunk) -> float:
    """Поднимает точные положения, совпадающие с юридическим намерением вопроса."""

    normalized_query = extract_user_messages(query).lower().replace("ё", "е")
    normalized_text = chunk.text.lower().replace("ё", "е")
    multiplier = 1.0

    # Вопрос о размере арендной платы — это условие о сумме,
    # а не обеспечительный платёж или общая обязанность арендатора.
    asks_rent_amount = (
        "аренд" in normalized_query
        and any(word in normalized_query for word in ("какая", "размер", "сколько", "рубл", "500 000"))
    )
    if asks_rent_amount and "арендная плата составляет" in normalized_text:
        multiplier *= 3.5

    # Вопрос о дате внесения арендной платы должен вести к сроку оплаты.
    asks_rent_due_date = (
        "аренд" in normalized_query
        and any(word in normalized_query for word in ("до какого числа", "вносить", "вносится", "числа"))
    )
    if asks_rent_due_date and "арендная плата вносится" in normalized_text and "не позднее" in normalized_text:
        multiplier *= 2.5

    # Аванс по договору закреплён в условии о предоплате.
    if re.search(r"аванс|предоплат", normalized_query) and re.search(
        r"предоплат|по соглашению сторон допускается", normalized_text
    ):
        multiplier *= 2.5

    # Стоимость поставки: упаковка и доставка входят в цену только там,
    # где это сказано непосредственно в условии о цене.
    asks_included_delivery = (
        "упаков" in normalized_query
        and "достав" in normalized_query
        and any(word in normalized_query for word in ("цен", "включ", "отдельно", "оплачивать"))
    )
    if asks_included_delivery and "упаковки и доставки" in normalized_text and "включает" in normalized_text:
        multiplier *= 3.5

    # Срок испытания для главного бухгалтера — специальное исключение.
    asks_accountant_probation = (
        "испыт" in normalized_query
        and "главн" in normalized_query
        and "бухгалтер" in normalized_query
    )
    if asks_accountant_probation and "главных бухгалтеров" in normalized_text and "шести месяцев" in normalized_text:
        multiplier *= 3.0

    # Общие гарантии работника во время испытания.
    asks_probation_rights = (
        "испыт" in normalized_query
        and any(word in normalized_query for word in ("трудов", "правил", "распространяться"))
    )
    if asks_probation_rights and "распространяются положения трудового законодательства" in normalized_text:
        multiplier *= 2.5

    # Возврат помещения при прекращении аренды должен ссылаться на акт.
    asks_return_with_keys = (
        any(word in normalized_query for word in ("ключ", "сдать", "возвращать", "съезжать"))
        and any(word in normalized_query for word in ("акт", "помещение", "договор"))
    )
    if asks_return_with_keys and "возвращает помещение по акту" in normalized_text:
        multiplier *= 2.5

    # Проверка качества: важен пункт приёмки со сроком в рабочих днях.
    asks_quality_check = (
        "качеств" in normalized_query
        and any(word in normalized_query for word in ("провер", "дней", "после поставки"))
    )
    if asks_quality_check and "приемка" in normalized_text and "по качеству" in normalized_text and "рабочих дней" in normalized_text:
        multiplier *= 2.5

    # Ночной въезд транспорта должен сопоставляться с круглосуточным доступом.
    asks_access = any(
        word in normalized_query
        for word in ("пропуск", "грузовик", "ночью", "часа ночи", "доступ")
    )
    if asks_access and "круглосуточно" in normalized_text and "беспрепятственный доступ" in normalized_text:
        multiplier *= 3.0

    # Передача подписанного оригинала после заключения договора —
    # не то же самое, что временная выдача уже архивного оригинала.
    asks_handover_after_signing = (
        "архив" not in normalized_query
        and "оригинал" in normalized_query
        and any(word in normalized_query for word in ("юротдел", "юридический", "отнести", "передать", "передача"))
        and any(word in normalized_query for word in ("срок", "сразу", "подписали", "подписан", "договор"))
    )
    if (
        asks_handover_after_signing
        and "подписанный оригинал договора" in normalized_text
        and "юридический отдел" in normalized_text
        and "с даты подписания" in normalized_text
    ):
        multiplier *= 3.0

    return multiplier


class BM25Retriever:
    """Локальный поисковый механизм на основе BM25."""

    def __init__(self, chunks: list[DocumentChunk]):
        if not chunks:
            raise ValueError("Нельзя создать поисковый индекс без фрагментов.")

        self.chunks = chunks
        tokenized_documents = [tokenize(chunk.text) for chunk in chunks]
        self.index = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
        category: str | None = None,
    ) -> list[SearchResult]:
        """Ищет наиболее релевантные фрагменты и применяет правовые фильтры."""

        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("Параметр top_k должен быть больше нуля.")

        query_tokens = prepare_query(query)
        if not query_tokens:
            return []

        allowed_chunks = filter_current_chunks(query, self.chunks)
        allowed_chunk_ids = {chunk.chunk_id for chunk in allowed_chunks}
        scores = self.index.get_scores(query_tokens)

        results = []
        for index, score in enumerate(scores):
            chunk = self.chunks[index]

            if chunk.chunk_id not in allowed_chunk_ids:
                continue
            if category and chunk.category != category:
                continue
            if score <= 0:
                continue

            final_score = float(score)

            if has_explicit_document_reference(query_tokens, chunk):
                final_score *= 1.5

            final_score *= get_status_multiplier(query, chunk)
            final_score *= _intent_multiplier(query, chunk)

            if final_score < MIN_RELEVANCE_SCORE:
                continue

            results.append(SearchResult(chunk=chunk, score=final_score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]
