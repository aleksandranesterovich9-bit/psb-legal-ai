"""Стеммер русского языка (алгоритм Snowball) без внешних зависимостей.

На платформе автоматической проверки нет pymorphy3, поэтому нормализация
словоформ реализована здесь: стеммер детерминирован, работает одинаково
локально, в Docker и на сервере организаторов.
"""

from __future__ import annotations

from functools import lru_cache

_VOWELS = "аеиоуыэюя"

_PERFECTIVE_GERUND_1 = ("вшись", "вши", "в")
_PERFECTIVE_GERUND_2 = ("ившись", "ывшись", "ивши", "ывши", "ив", "ыв")
_ADJECTIVE = (
    "ими", "ыми", "его", "ого", "ему", "ому",
    "ее", "ие", "ые", "ое", "ей", "ий", "ый", "ой", "ем", "им", "ым", "ом",
    "их", "ых", "ую", "юю", "ая", "яя", "ою", "ею",
)
_PARTICIPLE_1 = ("ем", "нн", "вш", "ющ", "щ")
_PARTICIPLE_2 = ("ивш", "ывш", "ующ")
_REFLEXIVE = ("ся", "сь")
_VERB_1 = (
    "ете", "йте", "ешь", "нно",
    "ла", "на", "ли", "ем", "ло", "но", "ет", "ют", "ны", "ть",
    "й", "л", "н",
)
_VERB_2 = (
    "ейте", "уйте",
    "ила", "ыла", "ена", "ите", "или", "ыли", "ило", "ыло", "ено", "ует",
    "уют", "ены", "ить", "ыть", "ишь",
    "ей", "уй", "ил", "ыл", "им", "ым", "ен", "ят", "ит", "ыт", "ую",
    "ю",
)
_NOUN = (
    "иями", "ями", "ами", "ией", "иям", "ием", "иях",
    "ев", "ов", "ие", "ье", "еи", "ии", "ей", "ой", "ий", "ям", "ем", "ам",
    "ом", "ах", "ях", "ию", "ью", "ия", "ья",
    "а", "е", "и", "й", "о", "у", "ы", "ь", "ю", "я",
)
_SUPERLATIVE = ("ейше", "ейш")
_DERIVATIONAL = ("ость", "ост")

# Существительные на «-ал»: классический Snowball принимает «л» за глагольное
# окончание («оригинал» -> «оригина»), из-за чего формы слова расходятся.
_NOUN_AL = (
    "оригинал", "филиал", "квартал", "материал", "персонал", "капитал",
    "журнал", "интервал", "терминал", "сигнал", "канал", "потенциал",
)


def _regions(word: str) -> tuple[int, int]:
    """Возвращает позиции начала областей RV и R2."""

    rv = len(word)
    for index, char in enumerate(word):
        if char in _VOWELS:
            rv = index + 1
            break

    def r_after(start: int) -> int:
        for index in range(start + 1, len(word)):
            if word[index] not in _VOWELS and word[index - 1] in _VOWELS:
                return index + 1
        return len(word)

    r1 = r_after(0)
    r2 = r_after(r1)
    return rv, r2


def _strip(word: str, rv: int, endings: tuple[str, ...], need_a_ya: bool = False) -> str | None:
    """Удаляет самое длинное окончание из набора, если оно лежит в области RV."""

    for ending in sorted(endings, key=len, reverse=True):
        if not word.endswith(ending):
            continue
        cut = len(word) - len(ending)
        if cut < rv:
            continue
        if need_a_ya:
            if cut - 1 < rv or word[cut - 1] not in "ая":
                continue
        return word[:cut]
    return None


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Приводит русское слово к основе. Латиница и числа не изменяются."""

    word = word.lower().replace("ё", "е")
    if len(word) < 3 or not any("а" <= char <= "я" for char in word):
        return word
    if any(char.isdigit() for char in word):
        return word
    for noun in _NOUN_AL:
        if word.startswith(noun) and len(word) - len(noun) <= 3:
            return noun

    rv, r2 = _regions(word)

    # Шаг 1
    result = _strip(word, rv, _PERFECTIVE_GERUND_2)
    if result is None:
        result = _strip(word, rv, _PERFECTIVE_GERUND_1, need_a_ya=True)
    if result is not None:
        word = result
    else:
        result = _strip(word, rv, _REFLEXIVE)
        if result is not None:
            word = result

        adjective = _strip(word, rv, _ADJECTIVE)
        if adjective is not None:
            word = adjective
            participle = _strip(word, rv, _PARTICIPLE_2)
            if participle is None:
                participle = _strip(word, rv, _PARTICIPLE_1, need_a_ya=True)
            if participle is not None:
                word = participle
        else:
            verb = _strip(word, rv, _VERB_2)
            if verb is None:
                verb = _strip(word, rv, _VERB_1, need_a_ya=True)
            if verb is not None:
                word = verb
            else:
                noun = _strip(word, rv, _NOUN)
                if noun is not None:
                    word = noun

    # Шаг 2
    if word.endswith("и") and len(word) - 1 >= rv:
        word = word[:-1]

    # Шаг 3
    for ending in _DERIVATIONAL:
        if word.endswith(ending) and len(word) - len(ending) >= r2:
            word = word[: -len(ending)]
            break

    # Шаг 4
    if word.endswith("нн") and len(word) - 1 >= rv:
        word = word[:-1]
    else:
        superlative = _strip(word, rv, _SUPERLATIVE)
        if superlative is not None:
            word = superlative
            if word.endswith("нн"):
                word = word[:-1]
        elif word.endswith("ь") and len(word) - 1 >= rv:
            word = word[:-1]

    return word
