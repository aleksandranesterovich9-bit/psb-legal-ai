"""Собирает dist/submission.zip для платформы: solution.py + пакет app (без веб-части)."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"api", "web", "__pycache__"}
target = ROOT / "dist" / "submission.zip"
target.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.write(ROOT / "solution.py", "solution.py")
    archive.writestr("requirements.txt", "httpx==0.28.1\n")
    for path in sorted((ROOT / "app").rglob("*.py")):
        if not SKIP & set(path.relative_to(ROOT).parts):
            archive.write(path, path.relative_to(ROOT).as_posix())
    names = archive.namelist()
print(f"{target} ({target.stat().st_size // 1024} КБ):", *names, sep="\n  ")
