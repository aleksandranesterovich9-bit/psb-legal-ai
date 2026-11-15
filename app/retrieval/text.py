"""Нормализация русского юридического текста: токены, синонимы, числа, даты, номера."""

from __future__ import annotations

import re
from datetime import date

from app.retrieval.stemmer import stem

_WORD_RE = re.compile(r"[a-zа-я0-9]+")

# Служебные слова не участвуют в поиске и оценке покрытия вопроса.
_STOPWORD_FORMS = """
и в во на по с со к ко у о об обо от до за из изо для при про над под без через между
или а но да же ли бы то не ни как так что чтобы это этот эта эти тот та те того
мы нас нам наш наша наше наши я мне меня мой он она оно они его ее их им ей вы вас вам ваш
какой какая какое какие каков сколько когда где куда кто чем кому чего почему зачем
можно нужно надо должен должна должны должно ли есть быть был была было были будет будут
если только уже еще ещё очень также тоже либо всё все всех весь вся там тут здесь
пожалуйста подскажите подскажи скажите скажи уточните значит хочу хотим хотел нужно ли
пользователь ассистент вопрос ответ например иногда просто вообще сразу какой-то
ооо ао пао зао рф г гг руб рублей рубля рубль
""".split()

STOPWORDS: frozenset[str] = frozenset({stem(word) for word in _STOPWORD_FORMS} | set(_STOPWORD_FORMS))

# Разговорные и профессиональные перефразировки -> термины, которыми пишут
# договоры, регламенты и законы. Это общий юридический словарь, а не
# привязка к конкретным вопросам: он расширяет лексический поиск.
_SYNONYM_SOURCE: dict[str, str] = {
    "аванс авансом авансовый заранее вперед": "предоплата",
    "предоплата": "аванс",
    "гарантия": "гарантийный срок",
    "брак бракованный дефект дефектный поломка сломанный неисправность некачественный": "недостаток качество",
    "проверить проверка осмотреть": "приемка",
    "съезжать съехать выезжать выехать освободить переезжать": "возврат помещение прекращение",
    "ключи ключ": "возврат помещение акт",
    "пропускать пропустить пускать впускать въезд проезд заезжать": "доступ",
    "ночью ночь ночной сутки": "круглосуточно",
    "грузовик фура машина автомобиль газель": "транспорт",
    "юротдел юрист юристы": "юридический отдел",
    "гендиректор гендир": "генеральный директор",
    "финдиректор": "финансовый директор",
    "главбух": "главный бухгалтер",
    "запустить запускать начать начинать инициировать": "инициатор",
    "отнести сдать отдать передавать передать": "передается",
    "штраф штрафовать санкция": "неустойка пени",
    "пеня пени неустойка": "штраф",
    "опоздание задержка задержать опоздать": "просрочка",
    "расторгнуть разорвать прекратить": "расторжение отказ исполнение",
    "платить заплатить оплатить оплачивать рассчитаться": "оплата расчеты",
    "стоить стоимость": "цена",
    "привезти привозить довезти": "доставка поставка",
    "испытательный": "испытание",
    "переработка переработки сверхурочные сверхурочно": "сверхурочная работа",
    "зарплата зп оклад": "заработная плата оплата труда",
    "минималка мрот": "минимальный размер оплаты труда",
    "ндс": "налог налоговая ставка добавленная стоимость",
    "ндфл подоходный": "налог доходы физических лиц налоговая ставка",
    "прибыль": "налоговая ставка",
    "пдн персданные": "персональные данные",
    "утечка слив": "неправомерная передача персональных данных",
    "зарубежный иностранец нерезидент": "иностранный контрагент",
    "санкции санкционный": "санкционная проверка",
    "поручительство поручиться": "поручитель",
    "залог заложить": "залогодатель",
    "связанный дочерний аффилированность": "аффилированные лица",
    "пересдать пересдавать": "субаренда",
    "коммуналка свет электричество": "коммунальные платежи электроэнергия",
    "депозит": "обеспечительный платеж",
    "поднять повысить поднимать повышать индексация индексировать": "увеличить",
    "суд судиться подсудность": "арбитражный суд споры",
    "форс мажор форсмажор": "обстоятельства непреодолимой силы",
    "упрощенка усн": "упрощенная система налогообложения",
    "сезон лето летний": "сезонные работы",
    "удалить стереть уничтожить": "прекратить обработка",
    "отозвать отзыв": "отозвано согласие",
    "одобрить одобрять утвердить согласовать": "одобрение согласование",
    "подписать подписали": "подписание подписанный",
    "хранить хранение": "архив хранится",
    "взять забрать получить": "выдача",
    "беременная декрет": "беременных женщин",
    "несовершеннолетний подросток": "не достигших возраста восемнадцати лет",
    "директор руководитель начальник": "руководитель",
    # Однокоренные формы, которые стеммер не сводит к одной основе.
    "арендный арендная арендатор арендодатель": "аренда",
    "аренда": "арендная",
    "гарантийный": "гарантия",
    "испытание": "испытательный",
    "налоговый налоговая": "налог",
    "налог": "налоговая",
    "складской": "склад",
    "поставщик поставлять поставить": "поставка",
    "покупать купить покупка закупка": "покупатель поставка",
    "согласовывать согласовать": "согласование",
    "персональный": "персональные данные",
    "трудовой": "труд работа",
    "претензионный": "претензия",
    "сверхурочный": "сверхурочная",
}


def normalize(text: str) -> str:
    """Нижний регистр, ё -> е, единообразные тире и кавычки."""

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return text


def words(text: str) -> list[str]:
    """Слова и числа без нормализации словоформ."""

    return _WORD_RE.findall(normalize(text))


def tokens(text: str) -> list[str]:
    """Основы всех слов текста (для индексации)."""

    return [stem(word) for word in words(text)]


def content_tokens(text: str) -> list[str]:
    """Основы значимых слов: без служебных и однобуквенных."""

    result = []
    for word in words(text):
        if word in STOPWORDS:
            continue
        token = stem(word)
        if token in STOPWORDS or (len(token) < 2 and not token.isdigit()):
            continue
        result.append(token)
    return result


def _build_synonyms() -> dict[str, tuple[str, ...]]:
    mapping: dict[str, list[str]] = {}
    for sources, targets in _SYNONYM_SOURCE.items():
        target_tokens = [stem(word) for word in targets.split()]
        for source in sources.split():
            bucket = mapping.setdefault(stem(source), [])
            for token in target_tokens:
                if token not in bucket and token not in STOPWORDS:
                    bucket.append(token)
    return {key: tuple(value) for key, value in mapping.items()}


SYNONYMS: dict[str, tuple[str, ...]] = _build_synonyms()


def expand_query(query_tokens: list[str], weight: float = 0.45) -> dict[str, float]:
    """Возвращает веса токенов запроса с добавлением юридических синонимов."""

    weights: dict[str, float] = {}
    for token in query_tokens:
        weights[token] = max(weights.get(token, 0.0), 1.0)
    for token in query_tokens:
        for synonym in SYNONYMS.get(token, ()):
            if synonym not in weights:
                weights[synonym] = weight
    return weights


# --------------------------------------------------------------------------
# Идентификаторы документов
# --------------------------------------------------------------------------

_LATIN_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")

# А-08/2024, П-17/2024, ПТ-05/2023, а также «П-17-2024» и «П17/2024».
_CONTRACT_NUMBER_RE = re.compile(
    r"(?<![A-Za-zА-Яа-яЁё0-9])([A-ZА-ЯЁ]{1,3})\s?[-‐‑‒–—]?\s?(\d{1,3})\s?[/\\‐‑‒–—-]\s?((?:19|20)\d{2})(?!\d)",
    re.IGNORECASE,
)
# К-12, ЮО-7, 14-П, 2-П — прочие регистрационные номера после знака №.
_REG_NUMBER_RE = re.compile(r"№\s*([A-ZА-ЯЁ0-9]{1,4}\s?[-‐‑‒–—]\s?[A-ZА-ЯЁ0-9]{1,4})(?![/\d])", re.IGNORECASE)


def canonical_number(prefix: str, number: str, year: str | None = None) -> str:
    """Единый вид номера договора: «А-08/2024»."""

    prefix = prefix.upper().translate(_LATIN_TO_CYRILLIC)
    number = number.zfill(2)
    return f"{prefix}-{number}/{year}" if year else f"{prefix}-{number}"


def loose_key(identifier: str) -> str:
    """Ключ сравнения без разделителей и ведущих нулей: «А-08/2024» -> «а82024»."""

    value = identifier.upper().translate(_LATIN_TO_CYRILLIC).lower()
    parts = re.findall(r"[а-яёa-z]+|\d+", value)
    return "".join(part.lstrip("0") or "0" if part.isdigit() else part for part in parts)


def find_contract_numbers(text: str) -> list[str]:
    """Номера договоров в каноническом виде, в порядке появления."""

    found: list[str] = []
    for match in _CONTRACT_NUMBER_RE.finditer(text):
        value = canonical_number(match.group(1), match.group(2), match.group(3))
        if value not in found:
            found.append(value)
    return found


def find_registration_numbers(text: str) -> list[str]:
    """Номера писем, записок и приказов вида «К-12», «ЮО-7», «14-П»."""

    found: list[str] = []
    for match in _REG_NUMBER_RE.finditer(text):
        raw = re.sub(r"\s", "", match.group(1)).upper().translate(_LATIN_TO_CYRILLIC)
        raw = re.sub(r"[‐‑‒–—]", "-", raw)
        if raw not in found:
            found.append(raw)
    return found


# --------------------------------------------------------------------------
# Даты
# --------------------------------------------------------------------------

MONTHS: dict[str, int] = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}
_MONTH_WORD = r"(январ[яеь]|феврал[яеь]|март[аеу]?|апрел[яеь]|ма[йяею]|июн[яеь]|июл[яеь]|август[аеу]?|сентябр[яеь]|октябр[яеь]|ноябр[яеь]|декабр[яеь])"

_NUMERIC_DATE_RE = re.compile(r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.((?:19|20)\d{2})(?!\d)")
_TEXT_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})(?:-?го)?\s+" + _MONTH_WORD + r"\s+((?:19|20)\d{2})", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(r"(?<![\d]\s)(?<!\d)\b" + _MONTH_WORD + r"\s+((?:19|20)\d{2})", re.IGNORECASE)


def month_number(word: str) -> int | None:
    word = normalize(word)
    for prefix, number in sorted(MONTHS.items(), key=lambda item: -len(item[0])):
        if word.startswith(prefix):
            return number
    return None


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[date, int, int]]:
    """Полные даты (число.месяц.год и «1 августа 2024») с позициями в тексте."""

    found: list[tuple[date, int, int]] = []
    for match in _NUMERIC_DATE_RE.finditer(text):
        value = _safe_date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        if value:
            found.append((value, match.start(), match.end()))
    for match in _TEXT_DATE_RE.finditer(text):
        month = month_number(match.group(2))
        if month is None:
            continue
        value = _safe_date(int(match.group(3)), month, int(match.group(1)))
        if value:
            found.append((value, match.start(), match.end()))
    found.sort(key=lambda item: item[1])
    return found


def find_month_years(text: str) -> list[tuple[int, int, int, int]]:
    """Упоминания «в августе 2024» без числа: (год, месяц, начало, конец)."""

    full_spans = [(start, end) for _, start, end in find_dates(text)]
    found = []
    for match in _MONTH_YEAR_RE.finditer(text):
        if any(start <= match.start() < end for start, end in full_spans):
            continue
        month = month_number(match.group(1))
        if month:
            found.append((int(match.group(2)), month, match.start(), match.end()))
    return found


def format_date(value: date | None) -> str:
    return value.strftime("%d.%m.%Y") if value else ""


# --------------------------------------------------------------------------
# Числа (для проверки ответа по источникам)
# --------------------------------------------------------------------------

_NUMBER_WORDS: dict[str, float] = {
    "ноль": 0, "нол": 0,
    "один": 1, "одн": 1, "перв": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "втор": 2, "двумя": 2,
    "три": 3, "трех": 3, "трем": 3, "трет": 3, "тремя": 3,
    "четыр": 4, "четверт": 4, "четырех": 4,
    "пят": 5, "шест": 6, "сем": 7, "седьм": 7, "восем": 8, "восьм": 8, "девят": 9,
    "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13,
    "четырнадцат": 14, "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17,
    "восемнадцат": 18, "девятнадцат": 19, "двадцат": 20, "тридцат": 30,
    "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесят": 50, "шестьдесят": 60,
    "шестидесят": 60, "семьдесят": 70, "семидесят": 70, "восемьдесят": 80,
    "восьмидесят": 80, "девяност": 90, "сто": 100, "ста": 100, "двест": 200,
    "трист": 300, "четырест": 400, "пятьсот": 500, "пятисот": 500,
    "полутор": 1.5, "полтор": 1.5, "двойн": 2, "тройн": 3,
}
_SCALE_WORDS: dict[str, float] = {
    "тыс": 1e3, "тысяч": 1e3, "тысяча": 1e3,
    "млн": 1e6, "миллион": 1e6,
    "млрд": 1e9, "миллиард": 1e9,
}
_DIGIT_NUMBER_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:[  ]\d{3})+|\d+)(?:[.,](\d+))?")


def _word_value(token: str) -> float | None:
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    stemmed = stem(token)
    if stemmed in _NUMBER_WORDS:
        return _NUMBER_WORDS[stemmed]
    for key, value in _NUMBER_WORDS.items():
        if len(key) >= 4 and token.startswith(key):
            return value
    return None


def _scale_value(token: str) -> float | None:
    for key, value in _SCALE_WORDS.items():
        if token == key or (len(key) > 3 and token.startswith(key)):
            return value
    return None


def extract_numbers(text: str) -> set[float]:
    """Все числовые значения текста: цифры, «12 млн», числительные словами.

    Даты исключаются (их сравнивает отдельная проверка), чтобы день и месяц
    не считались самостоятельными числовыми утверждениями.
    """

    text = normalize(text)
    for _, start, end in sorted(find_dates(text), key=lambda item: -item[1]):
        text = text[:start] + " " * (end - start) + text[end:]

    values: set[float] = set()

    for match in _DIGIT_NUMBER_RE.finditer(text):
        integer = re.sub(r"[  ]", "", match.group(1))
        fraction = match.group(2)
        # «13.11» и «2.4» как номера статей/пунктов отфильтровываются вызывающим кодом.
        value = float(f"{integer}.{fraction}") if fraction else float(integer)
        tail = text[match.end(): match.end() + 14].strip()
        tail_word = re.match(r"[а-я]+", tail)
        scale = _scale_value(tail_word.group(0)) if tail_word else None
        if scale:
            values.add(round(value * scale, 6))
        else:
            values.add(round(value, 6))

    word_list = _WORD_RE.findall(text)
    index = 0
    while index < len(word_list):
        token = word_list[index]
        value = _word_value(token) if not token.isdigit() else None
        if value is None:
            index += 1
            continue
        total = value
        cursor = index + 1
        # Составные числительные: «двадцать четыре», «сто двадцать».
        while cursor < len(word_list):
            next_value = _word_value(word_list[cursor]) if not word_list[cursor].isdigit() else None
            if next_value is None or next_value >= total or total % 10 != 0 or next_value != int(next_value):
                break
            total += next_value
            cursor += 1
        scale = _scale_value(word_list[cursor]) if cursor < len(word_list) else None
        if scale:
            total *= scale
            cursor += 1
        values.add(round(total, 6))
        index = cursor

    return values
