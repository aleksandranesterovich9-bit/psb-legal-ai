import re


# Убираем явные инструкции, адресованные модели, из недоверенного
# содержимого документов перед передачей в LLM или показом в fallback.
_AI_INSTRUCTION_LINE = re.compile(
    r"^\s*(?:ВНИМАНИЕ|ИНСТРУКЦИЯ)\s+ДЛЯ\s+(?:ИИ|AI|МОДЕЛИ|АССИСТЕНТА)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def sanitize_document_content(text: str) -> str:
    """Удаляет строки с явными prompt-injection инструкциями для ИИ."""

    cleaned = _AI_INSTRUCTION_LINE.sub(
        "[Строка с инструкцией для ИИ исключена как недоверенное содержимое.]",
        text,
    )
    return cleaned.strip()
