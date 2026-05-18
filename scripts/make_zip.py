"""
scripts/make_zip.py
Creates a clean zip of the project for uploading to Google Drive / Colab.
Excludes .git, results/, __pycache__, virtual envs, and log files.

Run from the repo root:
    python scripts/make_zip.py
"""

import zipfile
import os
import sys

EXCLUDE_DIRS = {
    ".git", "__pycache__", "results", ".venv", "venv", "env",
    ".eggs", "dist", "build", ".ipynb_checkpoints", "node_modules",
}
EXCLUDE_EXTS = {".pyc", ".pyo", ".log", ".json"}
EXCLUDE_FILES = {"aiisc.log"}

OUTPUT = "../peccavi_text.zip"   # one level up, outside the repo


def should_skip(path: str, name: str) -> bool:
    if name in EXCLUDE_FILES:
        return True
    _, ext = os.path.splitext(name)
    if ext in EXCLUDE_EXTS:
        return True
    return False


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_path = os.path.abspath(os.path.join(root, OUTPUT))

    print(f"Zipping  {root}")
    print(f"      →  {out_path}")
    print(f"Excluding: {sorted(EXCLUDE_DIRS)}")

    file_count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories in-place (affects os.walk recursion)
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for filename in filenames:
                if should_skip(dirpath, filename):
                    continue
                abs_path = os.path.join(dirpath, filename)
                # Store with a clean relative path: peccavi_text/<rest>
                rel = os.path.relpath(abs_path, os.path.dirname(root))
                zf.write(abs_path, arcname=rel)
                file_count += 1

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone — {file_count} files, {size_mb:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
