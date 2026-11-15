# PSB Legal AI

RAG-прототип юридического помощника для поиска по внутренним документам и нормативным материалам.

## Возможности

- Загрузка TXT, MD и DOCX из ZIP-архива базы знаний.
- Разбиение документов на структурные фрагменты с сохранением статей, частей и пунктов.
- Лексический поиск BM25 с русской лемматизацией и юридическими синонимами.
- Фильтрация утративших силу редакций и выбор редакции, действовавшей на указанную дату.
- Гибридный поиск BM25 + BGE-M3, объединение результатов через Reciprocal Rank Fusion (RRF).
- Повторное ранжирование через Reranker при наличии API.
- Генерация ответа через Chat Completions с проверкой ссылок на источники.
- Отказ от ответа при недостатке подтверждающих данных.
- Защита от явных prompt-injection строк внутри документов.
- Локальный режим BM25, если API-модели недоступны.

## Структура

```text
app/
  generation/       # генерация ответов и обработка недоверенного текста
  ingestion/        # загрузка документов и chunking
  retrieval/        # BM25, vector search, hybrid search, reranking
  llm_client.py     # клиент API embeddings/rerank/chat
  main.py           # FastAPI-приложение
 data/
  documents/knowledge-base.zip
  demo_questions.csv
  demo_output.csv
 solution.py         # официальный CSV-адаптер
 submission.csv      # последний локальный результат
 tests/              # тесты загрузки, поиска, дат и API-клиента
```

## Установка

Python 3.13 рекомендуется. В корне проекта:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Конфигурация моделей

Скопируйте `.env.example` в `.env` и заполните адрес API и выданные организаторами ключи. Не добавляйте `.env` в архивы и систему контроля версий.

```env
LLM_BASE_URL=
LLM_API_KEY=
EMBEDDER_API_KEY=
RERANKER_API_KEY=
LLM_MODEL=deepseek/deepseek-v4.1-flash
EMB_MODEL=baai/bge-m3
RERANKER_MODEL=cohere/rerank-5-fast
```

Если адрес API или ключи не заданы, проект не имитирует работу моделей: он использует BM25 и возвращает найденные выдержки с источниками. Генеративный ответ, embeddings и reranking в этом режиме недоступны.

## Официальный CSV-адаптер

```powershell
python solution.py
```

Адаптер использует `data/questions.csv` и `data/kb.zip`, если они переданы оценочной системой. Для локального запуска, если этих файлов нет, используются `data/demo_questions.csv` и `data/documents/knowledge-base.zip`. Результат сохраняется в `submission.csv` с колонками `id,answer`.

## FastAPI demo

Qdrant можно запустить через Docker Compose:

```powershell
docker compose up -d
```

Запуск приложения:

```powershell
uvicorn app.main:app --reload
```

Swagger UI: `http://127.0.0.1:8000/docs`.

1. Выполните `POST /index` — загрузка документов и построение индексов.
2. Выполните `POST /chat` с JSON-телом:

```json
{
  "question": "Какой срок проверки качества по договору П-17/2024?",
  "top_k": 5
}
```

Если API embeddings или Qdrant недоступны, индексирование переключается на BM25. Если API генерации недоступен, `/chat` возвращает найденные выдержки, а не выдуманный ответ.

## Тесты

```powershell
python -m pytest -q
```

Тесты проверяют загрузку и chunking, релевантность по демонстрационным вопросам, выбор редакции на историческую дату, abstain при отсутствии сведений, prompt-injection sanitization, API-клиент и векторную индексацию с mock-объектами.

## Поисковый конвейер

```text
ZIP documents
      ↓
Document Loader → Chunker
      ↓
 BM25 ─────── BGE-M3
      │            │
      └── RRF ─────┘
           ↓
        Reranker
           ↓
     Legal Answer Generator
           ↓
   Answer + source references
```

Для юридической точности BM25 используется как evidence gate: если лексический поиск не находит подтверждающих фрагментов, векторная близость сама по себе не создаёт ответ.
