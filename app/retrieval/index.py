"""Гибридный поиск по базе знаний: BM25 + BGE-M3 (RRF) + реранкер + юридические метаданные.

Работает без внешних сервисов (только BM25); эмбеддинги и реранкер
подключаются, если доступен API, и отключаются при сбое без потери ответа.
"""

from __future__ import annotations

import calendar
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import date

from app.generation.security import detect_user_injection, looks_unlawful
from app.ingestion.chunker import Chunk, chunk_document
from app.ingestion.document_loader import SourceDocument, load_documents_from_zip
from app.retrieval import text as tx

logger = logging.getLogger(__name__)

RRF_K = 60
RERANK_CANDIDATES = 24
_ROLE_RE = re.compile(
    r"^\s*(Пользователь|Юрист|Клиент|Сотрудник|User|Human|Ассистент|Бот|Assistant|AI|ИИ)\s*:\s*", re.I | re.M
)
_ASSISTANT_ROLES = {"ассистент", "бот", "assistant", "ai", "ии"}
_HISTORICAL_RE = re.compile(r"редакци|утратил|прежн|стар(?:ая|ой|ую|ом)\b|раньше|ранее|изменил|было\s+до|на\s+тот\s+момент|действовал", re.I)
_GENERIC_SUBJECT = set(tx.tokens("поставщик покупатель обязуется передать принять оплатить товар далее основании заявок собственность условиях настоящего договора поставлять принимать оплачивать арендодатель арендатор передает"))


@dataclass
class Query:
    raw: str
    turns: list[tuple[str, str]]            # (user|assistant, текст)
    question: str                           # последняя реплика пользователя
    context: str = ""                       # предыдущие реплики
    targets: list[str] = field(default_factory=list)     # doc_id явно названных документов
    ambiguous: list[str] = field(default_factory=list)   # doc_id кандидатов при неоднозначности
    period: tuple[date, date] | None = None              # период события
    historical: bool = False
    injection: list[str] = field(default_factory=list)
    unlawful: bool = False
    missing_terms: list[str] = field(default_factory=list)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rerank: float | None = None


def parse_dialogue(raw: str) -> list[tuple[str, str]]:
    """Разбирает «Пользователь: ... Ассистент: ...»; одиночный вопрос — одна реплика."""

    matches = list(_ROLE_RE.finditer(raw))
    if not matches or matches[0].start() > 0 and raw[: matches[0].start()].strip():
        return [("user", raw.strip())]
    turns = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        role = "assistant" if match.group(1).lower() in _ASSISTANT_ROLES else "user"
        text = raw[match.end():end].strip()
        if text:
            turns.append((role, text))
    return turns or [("user", raw.strip())]


def extract_period(text: str) -> tuple[date, date] | None:
    """Дата или период события из текста вопроса."""

    dates = tx.find_dates(text)
    if dates:
        return dates[0][0], dates[0][0]
    months = tx.find_month_years(text)
    if months:
        year, month = months[0][0], months[0][1]
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    year = re.search(r"\b(?:в|за|на|с|до|от)\s+((?:19|20)\d{2})\s*(?:году|года|год|г\.|г\b)", tx.normalize(text))
    if year:
        return date(int(year.group(1)), 1, 1), date(int(year.group(1)), 12, 31)
    return None


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.4, b: float = 0.6):
        self.k1, self.b = k1, b
        self.lengths = [len(tokens) for tokens in documents]
        self.avg = sum(self.lengths) / max(1, len(documents))
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(documents):
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                self.postings.setdefault(token, []).append((index, count))
        total = len(documents)
        self.idf = {t: math.log(1 + (total - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, weights: dict[str, float]) -> list[float]:
        result = [0.0] * len(self.lengths)
        for token, weight in weights.items():
            for index, count in self.postings.get(token, ()):
                norm = count + self.k1 * (1 - self.b + self.b * self.lengths[index] / self.avg)
                result[index] += weight * self.idf[token] * count * (self.k1 + 1) / norm
        return result


class KnowledgeBase:
    def __init__(self, documents: list[SourceDocument], ai=None, today: date | None = None):
        self.documents = documents
        self.docs = {document.doc_id: document for document in documents}
        self.chunks: list[Chunk] = []
        self.doc_chunks: dict[str, list[Chunk]] = {}
        for document in documents:
            chunks = chunk_document(document)
            self.doc_chunks[document.doc_id] = chunks
            self.chunks.extend(chunks)
        self.bm25 = BM25([tx.tokens(chunk.search_text) for chunk in self.chunks])
        self.ai = ai
        self.today = today or date.today()
        self.vectors: list[list[float]] | None = None
        self.search_mode = "bm25"
        self._names = self._build_names()

    @classmethod
    def from_zip(cls, path, ai=None) -> "KnowledgeBase":
        return cls(load_documents_from_zip(path), ai=ai)

    # ------------------------------------------------------------------ индекс
    def build_vectors(self) -> bool:
        """Считает эмбеддинги фрагментов; при сбое остаётся BM25."""

        if self.ai is None or not self.ai.has_embeddings:
            return False
        try:
            self.vectors = self.ai.embed([chunk.search_text[:2000] for chunk in self.chunks])
            self.search_mode = "hybrid"
            return True
        except Exception as error:  # noqa: BLE001 — любой сбой эмбеддера не критичен
            logger.warning("Эмбеддинги недоступны (%s); используется BM25.", error)
            self.vectors = None
            return False

    def _build_names(self) -> list[tuple[list[str], str]]:
        names = []
        for document in self.documents:
            for name in document.counterparties:
                inner = re.search(r"«(.+)»", name)
                tokens = tx.tokens(inner.group(1) if inner else name)
                if tokens:
                    names.append((tokens, document.doc_id))
        return names

    # --------------------------------------------------------------- сущности
    def _match_entities(self, text: str) -> tuple[list[str], list[str]]:
        """Возвращает (точно названные документы, кандидаты по неполному названию)."""

        exact: list[str] = []
        keys = {tx.loose_key(number) for number in tx.find_contract_numbers(text)}
        keys |= {tx.loose_key(number) for number in tx.find_registration_numbers(text)}
        for document in self.documents:
            if document.number and document.doc_type != "amendment" and tx.loose_key(document.number) in keys:
                exact.append(document.doc_id)

        tokens = tx.tokens(text)
        capitalized = {tx.stem(word) for word in re.findall(r"(?<![.!?]\s)\b[А-ЯЁ][а-яё]+|«\s*([А-ЯЁа-яё]+)", text) if word} | {
            tx.stem(word) for word in re.findall(r"\b[А-ЯЁ][а-яё]+", text)
        }
        partial: list[str] = []
        for name_tokens, doc_id in self._names:
            size = len(name_tokens)
            if any(tokens[i:i + size] == name_tokens for i in range(len(tokens) - size + 1)):
                exact.append(doc_id)
            elif size > 1 and name_tokens[0] in tokens and name_tokens[0] in capitalized:
                partial.append(doc_id)
        exact = list(dict.fromkeys(exact))
        # Доп. соглашения следуют за своим договором и отдельной целью не считаются.
        exact = [d for d in exact if self.docs[d].doc_type != "amendment" or not any(l in exact for l in self.docs[d].links)]
        partial = [d for d in dict.fromkeys(partial) if d not in exact and self.docs[d].doc_type != "amendment"]
        return exact, ([] if exact else partial)

    def _disambiguate(self, candidates: list[str], text: str) -> list[str]:
        """Выбирает кандидата по предмету договора («мебель», «компьютеры»)."""

        query_tokens = set(tx.content_tokens(text))
        scored = []
        for doc_id in candidates:
            subject = set(tx.content_tokens(self.docs[doc_id].subject or "")) - _GENERIC_SUBJECT
            scored.append((len(subject & query_tokens), doc_id))
        best = max(score for score, _ in scored)
        winners = [doc_id for score, doc_id in scored if score == best]
        return winners if best > 0 and len(winners) == 1 else []

    def analyze(self, raw: str) -> Query:
        turns = parse_dialogue(raw)
        user_turns = [text for role, text in turns if role == "user"]
        question = user_turns[-1] if user_turns else raw.strip()
        last_index = max((i for i, (role, _) in enumerate(turns) if role == "user"), default=0)
        previous = turns[:last_index]
        query = Query(raw=raw, turns=turns, question=question, context=" ".join(text for _, text in previous))

        exact, partial = self._match_entities(question)
        if not exact and not partial:
            for _, text in reversed(previous):
                exact, partial = self._match_entities(text)
                if exact or partial:
                    break
        if not exact and len(partial) == 1:
            exact, partial = partial, []
        if not exact and len(partial) > 1:
            resolved = self._disambiguate(partial, " ".join(text for role, text in turns if role == "user"))
            exact, partial = (resolved, []) if resolved else ([], partial)
        query.targets, query.ambiguous = exact, partial

        user_text = " ".join(user_turns)
        query.period = extract_period(question) or extract_period(user_text)
        query.historical = bool(_HISTORICAL_RE.search(question)) or (
            query.period is not None and query.period[1] < self.today.replace(day=1)
        )
        query.injection = detect_user_injection(user_text)
        query.unlawful = looks_unlawful(question)
        skip = {token for tokens, _ in self._names for token in tokens}
        # Существительные вопроса, которых (с учётом синонимов) нет в базе, — сигнал отказа.
        verb_like = re.compile(r"(?:ть|ться|тся|ем|им|ете|ите|ет|ит|ут|ют|ат|ят|ал|ала|али|ил|ила|или|ишь|ешь|у|ю)$")
        query.missing_terms = list(dict.fromkeys(
            word for word in tx.words(question)
            if word not in tx.STOPWORDS and len(word) > 4 and not word.isdigit() and not verb_like.search(word)
            and tx.stem(word) not in tx.STOPWORDS and tx.stem(word) not in self.bm25.postings and tx.stem(word) not in skip
            and not any(synonym in self.bm25.postings for synonym in tx.SYNONYMS.get(tx.stem(word), ()))
        ))
        return query

    def coverage(self, query: Query, chunks: list[Chunk]) -> float:
        """Доля (по IDF) значимых слов вопроса, найденных в данных фрагментах."""

        weights = tx.content_tokens(query.question)
        present = {token for chunk in chunks for token in tx.tokens(chunk.search_text)}
        top = max(self.bm25.idf.values())
        total = found = 0.0
        for token in dict.fromkeys(weights):
            idf = self.bm25.idf.get(token, top)
            total += idf
            if token in present or any(synonym in present for synonym in tx.SYNONYMS.get(token, ())):
                found += idf
        return found / total if total else 0.0

    # ------------------------------------------------------------------ поиск
    def _doc_multiplier(self, document: SourceDocument, query: Query) -> float:
        value = 1.0
        if query.targets:
            if document.doc_id in query.targets:
                value *= 3.0
            elif any(link in query.targets for link in document.links):
                value *= 2.0
            elif document.doc_type in {"contract", "amendment"}:
                value *= 0.15
        elif query.ambiguous and document.doc_id in query.ambiguous:
            value *= 2.0
        if document.family:
            if query.period:
                effective = document.effective_on(query.period[0]) or document.effective_on(query.period[1])
                value *= 1.5 if effective else 0.4
            elif document.status == "repealed":
                value *= 1.0 if query.historical else 0.2
        return value

    def search(self, query: Query, top_k: int = 30) -> list[Hit]:
        weights = tx.expand_query(tx.content_tokens(query.question))
        for token in tx.content_tokens(query.context):
            weights.setdefault(token, 0.3)
        article = re.search(r"\bст(?:ать[яиеюй]|\.)\s*(\d+(?:\.\d+)*)", tx.normalize(query.question))

        lexical = self.bm25.scores(weights)
        rankings = [sorted((i for i, s in enumerate(lexical) if s > 0), key=lambda i: -lexical[i])[:200]]

        if self.vectors is not None and self.ai.has_embeddings:
            try:
                text = query.question if not query.context else f"{query.context[-500:]}\n{query.question}"
                vector = self.ai.embed([text])[0]
                dense = [sum(a * b for a, b in zip(vector, row)) for row in self.vectors]
                rankings.append(sorted(range(len(dense)), key=lambda i: -dense[i])[:200])
            except Exception as error:  # noqa: BLE001
                logger.warning("Векторный поиск не выполнен (%s); используется BM25.", error)

        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (RRF_K + rank)
        top_lexical = lexical[rankings[0][0]] if rankings[0] else 1.0
        hits: list[Hit] = []
        for index, score in fused.items():
            chunk = self.chunks[index]
            document = self.docs[chunk.doc_id]
            # RRF даёт порядок, нормированный BM25 — силу лексического совпадения.
            score = score * (1.0 + lexical[index] / top_lexical) * self._doc_multiplier(document, query)
            if article and document.law_article == article.group(1):
                score *= 2.0
            if chunk.kind == "preamble" or tx.normalize(chunk.text).startswith(("поставщик:", "покупатель:", "арендодатель:")):
                score *= 0.6
            hits.append(Hit(chunk, score))
        hits.sort(key=lambda hit: -hit.score)
        hits = hits[:top_k]

        if self.ai is not None and self.ai.has_reranker and hits:
            head = hits[:RERANK_CANDIDATES]
            try:
                order = self.ai.rerank(query.question, [hit.chunk.search_text[:1500] for hit in head], top_n=len(head))
                if order:
                    reranked = []
                    for position, (index, relevance) in enumerate(order):
                        hit = head[index]
                        hit.rerank = relevance
                        # Реранкер задаёт порядок, но явно названный документ не теряется.
                        hit.score = (1.0 / (position + 1)) * self._doc_multiplier(self.docs[hit.chunk.doc_id], query)
                        reranked.append(hit)
                    seen = {id(hit) for hit in reranked}
                    reranked.sort(key=lambda hit: -hit.score)
                    floor = min(hit.score for hit in reranked)
                    rest = [hit for hit in hits if id(hit) not in seen]
                    for offset, hit in enumerate(rest, start=1):
                        hit.score = floor / (1 + offset)
                    hits = reranked + rest
                    self.search_mode = "hybrid+rerank" if self.vectors is not None else "bm25+rerank"
            except Exception as error:  # noqa: BLE001
                logger.warning("Реранкер недоступен (%s); сохраняется порядок гибридного поиска.", error)
        return hits

    # --------------------------------------------------------- отбор доказательств
    def select_evidence(self, query: Query, hits: list[Hit], max_chars: int = 30000) -> list[Chunk]:
        """Формирует набор источников: небольшие документы целиком, большие законы — лучшими фрагментами."""

        doc_scores: dict[str, float] = {}
        for hit in hits[:20]:
            doc_scores[hit.chunk.doc_id] = doc_scores.get(hit.chunk.doc_id, 0.0) + hit.score
        ordered = sorted(doc_scores, key=lambda doc_id: -doc_scores[doc_id])
        best = doc_scores[ordered[0]] if ordered else 0.0

        chosen: list[str] = list(query.targets or query.ambiguous)
        company = [d for d in ordered if not self.docs[d].is_law and doc_scores[d] >= 0.3 * best]
        for doc_id in company[:3]:
            if doc_id not in chosen:
                chosen.append(doc_id)
        for doc_id in list(chosen):
            for link in self.docs[doc_id].links:
                linked = self.docs[link]
                if linked.status == "repealed" and not (query.period or query.historical):
                    continue
                if link not in chosen:
                    chosen.append(link)
        if not (query.period or query.historical):
            chosen = [d for d in chosen if self.docs[d].status != "repealed" or d in query.targets]

        law_hits = [hit for hit in hits[:20] if self.docs[hit.chunk.doc_id].is_law][:8]
        law_selected: dict[str, set[int]] = {}
        for hit in law_hits:
            if doc_scores.get(hit.chunk.doc_id, 0.0) < 0.2 * best:
                continue
            chunks = self.doc_chunks[hit.chunk.doc_id]
            selected = law_selected.setdefault(hit.chunk.doc_id, set())
            if sum(len(chunk.text) for chunk in chunks) <= 7000:
                selected.update(range(len(chunks)))
            else:
                selected.add(hit.chunk.index)
                # Вводная часть пункта нужна, чтобы подпункт не отрывался от нормы.
                for chunk in chunks:
                    if chunk.number == hit.chunk.number and chunk.kind == hit.chunk.kind and chunk.lead is None:
                        selected.add(chunk.index)
                        break

        evidence: list[Chunk] = []
        total = 0
        doc_order = sorted(set(chosen) | set(law_selected), key=lambda d: (d not in query.targets, -doc_scores.get(d, 0.0)))
        for doc_id in doc_order:
            chunks = self.doc_chunks[doc_id] if doc_id in chosen else [self.doc_chunks[doc_id][i] for i in sorted(law_selected[doc_id])]
            size = sum(len(chunk.text) for chunk in chunks)
            if evidence and total + size > max_chars:
                continue
            evidence.extend(chunks)
            total += size
        return evidence
