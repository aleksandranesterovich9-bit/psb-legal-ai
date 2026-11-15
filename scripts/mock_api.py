"""Локальная имитация API организаторов (/chat/completions, /embeddings, /rerank).

Нужна только для проверки интеграции без ключей: «модель» отвечает дословной
выдержкой из самого похожего источника. Качество ответов она не отражает.
Запуск: python scripts/mock_api.py 8099
"""
import hashlib
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORD = re.compile(r"[a-zа-яё0-9]{3,}")


def words(text):
    return WORD.findall(text.lower())


def embed(text, size=256):
    vector = [0.0] * size
    for word in words(text):
        vector[int(hashlib.md5(word[:6].encode()).hexdigest(), 16) % size] += 1.0
    return vector


def chat(messages):
    prompt = messages[1]["content"] if len(messages) > 1 else ""
    sources = re.findall(r"^\[(S\d+)\] ([^:\n]*): (.+)$", prompt, re.M)
    turns = re.findall(r"^Пользователь: (.+)$", prompt, re.M)
    question = set(words(turns[-1] if turns else ""))
    best = max(sources, key=lambda s: len(question & set(words(s[2]))), default=None)
    if best is None or not question & set(words(best[2])):
        return {"status": "insufficient", "answer": "В базе знаний нет сведений по этому вопросу.", "citations": []}
    return {"status": "answer", "answer": f"По документу ({best[1]}): {best[2]}", "citations": [{"source": best[0], "quote": best[2][:120]}]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.endswith("/embeddings"):
            data = {"data": [{"index": i, "embedding": embed(t)} for i, t in enumerate(body["input"])]}
        elif self.path.endswith("/rerank"):
            query = set(words(body["query"]))
            scored = sorted(((len(query & set(words(d))) / (len(query) or 1), i) for i, d in enumerate(body["documents"])), reverse=True)
            data = {"results": [{"index": i, "relevance_score": s} for s, i in scored[: body.get("top_n", 10)]]}
        elif self.path.endswith("/chat/completions"):
            data = {"choices": [{"message": {"role": "assistant", "content": json.dumps(chat(body["messages"]), ensure_ascii=False)}}]}
        else:
            self.send_error(404)
            return
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"mock API: http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
