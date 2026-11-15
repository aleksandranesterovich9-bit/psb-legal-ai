"""Собирает dist/submission.zip для платформы.

Требование платформы: solution.py и все модули лежат в корне архива (без вложенных папок),
вопросы и база знаний в архив не входят. Пакет app разворачивается в плоские модули psb_*.py,
импорты переписываются автоматически.
"""
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"api", "web", "__pycache__"}
target = ROOT / "dist" / "submission.zip"
target.parent.mkdir(exist_ok=True)

modules = {}  # app.retrieval.index -> psb_index
sources = {}
for path in sorted((ROOT / "app").rglob("*.py")):
    parts = path.relative_to(ROOT).with_suffix("").parts
    if SKIP & set(parts) or path.stat().st_size == 0:
        continue
    modules[".".join(parts)] = "psb_" + parts[-1]
    sources[parts] = path

def flatten(code: str) -> str:
    # from app.retrieval import text as tx  ->  import psb_text as tx
    code = re.sub(r"^(\s*)from app\.\w+ import (\w+)( as \w+)?$", lambda m: f"{m[1]}import psb_{m[2]}{m[3] or ''}", code, flags=re.M)
    # from app.retrieval.index import X  ->  from psb_index import X
    code = re.sub(r"\bfrom app\.(?:\w+\.)*(\w+) import ", r"from psb_\1 import ", code)
    return code

with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(ROOT / "solution.py", "solution.py")
    archive.writestr("requirements.txt", "httpx==0.28.1\n")
    for parts, path in sources.items():
        code = flatten(path.read_text(encoding="utf-8"))
        if re.search(r"\bapp\.(?:retrieval|ingestion|generation|llm_client)", code.replace("(app.retrieval.index)", "")):
            leftover = re.findall(r"^.*\bapp\.(?:retrieval|ingestion|generation|llm_client).*$", code, flags=re.M)
            leftover = [line for line in leftover if not line.lstrip().startswith(("#", "A.", "B.", "C.", "D.")) and '"""' not in line]
            if leftover:
                raise SystemExit(f"Остались импорты app.*: {leftover}")
        archive.writestr(modules[".".join(parts)].replace("psb_llm_client", "psb_llm_client") + ".py", code)
    names = archive.namelist()
assert not any("/" in name for name in names), "в архиве вложенные папки"
print(f"{target} ({target.stat().st_size // 1024} КБ):", *names, sep="\n  ")
