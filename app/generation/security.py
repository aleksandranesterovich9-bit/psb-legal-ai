"""Защита от prompt-injection: недоверенный текст документов и реплик пользователя.

Принципы:
* текст документов — данные, а не инструкции: фрагменты, адресованные ИИ,
  вырезаются ещё при загрузке, чтобы не попасть ни в индекс, ни в промпт;
* реплики пользователя не изменяются, но попытки сменить роль, подделать
  системное сообщение или выведать промпт помечаются и передаются модели
  как сигнал, а ответ дополнительно проверяется на утечку инструкций.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

REMOVED_MARKER = "[Фрагмент удалён: содержал инструкции для ИИ-ассистента, а не положения документа.]"

_AI_WORD = r"(?:ии|ai|ии-ассистент\w*|ассистент\w*|модел\w*|нейросет\w*|чат-?бот\w*|бот\w*|llm|gpt|chatgpt)"

# Признаки инструкции, адресованной модели, внутри документа.
_DOCUMENT_INJECTION_PATTERNS = [
    re.compile(rf"(?:внимани\w*|инструкци\w*|указани\w*|команд\w*|сообщени\w*|примечани\w*)\s+(?:для|к)\s+{_AI_WORD}", re.I),
    re.compile(rf"(?:^|[\s«\"(]){_AI_WORD}\s*[:,—-]\s*(?:игнорируй|отвечай|забудь|выполни|сообщи|не упоминай)", re.I),
    re.compile(r"\b(?:игнорируй\w*|проигнорируй\w*|забудь|не учитывай|отмени)\b.{0,80}\b(?:инструкци\w*|документ\w*|правил\w*|указани\w*|контекст\w*|источник\w*|баз\w* знаний|огранич\w*)", re.I),
    re.compile(r"\bignore\b.{0,40}\b(?:previous|above|all|other|prior)\b.{0,40}\b(?:instructions?|documents?|rules?|context|prompts?)", re.I),
    re.compile(r"\bdisregard\b.{0,60}\b(?:instructions?|rules?|context)", re.I),
    re.compile(r"не\s+(?:упоминай|сообщай|раскрывай)\w*.{0,60}(?:инструкци|указани|сообщени|пользовател)", re.I),
    re.compile(r"на\s+(?:любой|каждый|все)\s+вопрос\w*.{0,160}\bотвечай", re.I),
    re.compile(r"\b(?:system prompt|системн\w+\s+(?:промпт|промт|инструкци\w*|сообщени\w*))", re.I),
    re.compile(r"\b(?:ты теперь|отныне ты|you are now|act as|веди себя как|представь,? что ты)\b", re.I),
]

# Признаки атаки в сообщении пользователя: (метка, шаблон).
_USER_INJECTION_PATTERNS = [
    ("подмена инструкций", re.compile(r"\b(?:игнорируй\w*|проигнорируй\w*|забудь|отмени|не учитывай|сбрось)\b.{0,80}\b(?:инструкци\w*|правил\w*|указани\w*|огранич\w*|промпт\w*|промт\w*|настройк\w*)", re.I)),
    ("подмена инструкций", re.compile(r"\b(?:ignore|disregard|forget|override)\b.{0,60}\b(?:instructions?|rules?|prompts?|guidelines|restrictions)", re.I)),
    ("поддельное системное сообщение", re.compile(r"(?:^|\n)\s*(?:[\[<#*]{0,3}\s*)?(?:system|системн\w+\s+сообщени\w+|сообщение\s+системы|администратор|developer|разработчик|admin|root)\s*(?:[\]>*]{0,3})\s*[:—-]", re.I)),
    ("поддельное системное сообщение", re.compile(r"<\s*/?\s*(?:system|sys|instructions?)\s*>|\[\s*/?\s*(?:system|inst)\s*\]", re.I)),
    ("смена роли", re.compile(r"\b(?:ты теперь|отныне ты|ты больше не|you are now|act as|веди себя как|притворись|представь,? что ты|режим разработчика|developer mode|jailbreak|\bdan\b)", re.I)),
    ("запрос системного промпта", re.compile(r"(?:покажи|выведи|напиши|раскрой|повтори|процитируй|перескажи|print|show|reveal|repeat).{0,60}(?:системн\w+\s+(?:промпт|промт|инструкци\w*|сообщени\w*)|system prompt|свои\s+инструкци\w*|твои\s+инструкци\w*|начальн\w+\s+инструкци\w*|скрыт\w+\s+инструкци\w*)", re.I)),
    ("требование отвечать без источников", re.compile(r"(?:без\s+(?:ссылок|источников|цитат)|не\s+(?:ссылайся|указывай\s+источник)|своими\s+знаниями|из\s+(?:своей\s+)?памяти|по\s+памяти|(?:используй|возьми)\s+(?:интернет|общие знания))", re.I)),
    ("требование выдумать ответ", re.compile(r"(?:придумай|выдумай|сочини|предположи)\b.{0,60}(?:норм\w*|стать\w*|пункт\w*|закон\w*|услови\w*|ответ\w*|документ\w*)", re.I)),
]

# Признаки просьбы помочь с нарушением (сигнал для модели, не автоматический отказ).
_UNLAWFUL_PATTERNS = [
    re.compile(r"\bкак\s+(?:нам\s+|мне\s+)?(?:обойти|избежать\s+ответственности|уклониться|скрыть|спрятать|замаскировать|не\s+платить)", re.I),
    re.compile(r"\b(?:задним\s+числом|подделать|подделк\w*|фиктивн\w*|сфальсифицировать|фальсифик\w*|подложн\w*)", re.I),
    re.compile(r"\bчтобы\s+(?:никто\s+)?не\s+(?:узнал\w*|заметил\w*|проверил\w*|нашл\w*)", re.I),
    re.compile(r"\b(?:обойти|обход)\s+(?:закон\w*|санкци\w*|проверк\w*|комплаенс\w*|регламент\w*|требовани\w*|валютн\w+\s+контрол\w*)", re.I),
    re.compile(r"\b(?:уйти|уход)\s+от\s+(?:налог\w*|штраф\w*|ответственност\w*)", re.I),
    re.compile(r"\bбез\s+(?:согласования|одобрения|проверки)\b.{0,60}\bчтобы\b", re.I),
]

_PARAGRAPH_SPLIT = re.compile(r"(?<=[.!?»\"])\s+(?=[А-ЯЁA-Z«\"])")


@dataclass
class SanitizedText:
    text: str
    removed: list[str] = field(default_factory=list)


def _is_document_injection(sentence: str) -> bool:
    normalized = sentence.replace("ё", "е").replace("Ё", "Е")
    return any(pattern.search(normalized) for pattern in _DOCUMENT_INJECTION_PATTERNS)


def sanitize_document_text(text: str) -> SanitizedText:
    """Удаляет из документа фрагменты, адресованные ИИ.

    Обработка идёт по абзацам: с первого предложения-инструкции и до конца
    абзаца текст заменяется маркером, остальные положения сохраняются.
    """

    removed: list[str] = []
    result_lines: list[str] = []

    for line in text.split("\n"):
        if not line.strip() or not _is_document_injection(line):
            result_lines.append(line)
            continue

        sentences = _PARAGRAPH_SPLIT.split(line)
        kept: list[str] = []
        dropped: list[str] = []
        triggered = False
        for sentence in sentences:
            if not triggered and _is_document_injection(sentence):
                triggered = True
            (dropped if triggered else kept).append(sentence)

        if not dropped:
            # Шаблон сработал только на абзац целиком — удаляем абзац.
            dropped, kept = sentences, []

        removed.append(" ".join(dropped).strip())
        kept_text = " ".join(kept).strip()
        result_lines.append(f"{kept_text} {REMOVED_MARKER}".strip())

    return SanitizedText(text="\n".join(result_lines), removed=removed)


def detect_user_injection(text: str) -> list[str]:
    """Возвращает метки обнаруженных попыток манипуляции в сообщении пользователя."""

    normalized = text.replace("ё", "е").replace("Ё", "Е")
    labels: list[str] = []
    for label, pattern in _USER_INJECTION_PATTERNS:
        if label not in labels and pattern.search(normalized):
            labels.append(label)
    return labels


def looks_unlawful(text: str) -> bool:
    """Эвристика: просьба помочь обойти требования, а не разъяснить их."""

    normalized = text.replace("ё", "е").replace("Ё", "Е")
    return any(pattern.search(normalized) for pattern in _UNLAWFUL_PATTERNS)


def escape_untrusted(text: str) -> str:
    """Не даёт недоверенному тексту закрыть служебные теги промпта."""

    return re.sub(r"<\s*(/?)\s*(источники|документ|диалог|вопрос|sources|document|system)\b", r"<\1_\2", text, flags=re.I)


def leaks_system_prompt(answer: str, canaries: tuple[str, ...]) -> bool:
    """Проверяет, не воспроизводит ли ответ характерные фразы системного промпта."""

    normalized = re.sub(r"\s+", " ", answer.lower())
    return any(re.sub(r"\s+", " ", canary.lower()) in normalized for canary in canaries)
