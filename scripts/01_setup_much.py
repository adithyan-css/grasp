#!/usr/bin/env python3
"""
GRASP stage 0 — set up the MUCH benchmark so it actually imports.

Three things bite on a fresh Colab/Kaggle runtime and this script handles all of
them:

1. **Missing dependencies.** `src/__init__.py` imports every submodule, so an
   import of anything pulls in the annotation and generation stacks too. The
   full import-time set is installed here. Torch/transformers/accelerate are
   deliberately NOT pinned, so the CUDA build already on the runtime survives.

2. **Python version.** MUCH targets Python 3.12. Two f-strings in
   `src/annotation/human_evaluator.py` put a backslash inside the expression
   part, which is PEP 701 and only legal on 3.12+. On a 3.11 runtime (common on
   Kaggle) the whole package fails to import with a SyntaxError pointing at a
   file you never intended to use. The patch below hoists those expressions out
   of the f-strings; behaviour is unchanged and it is a no-op on 3.12.

3. **Command names.** The baseline signals are imported with
   `python -m src import-signals`, not `import-baselines`.

Usage
-----
    python scripts/01_setup_much.py --root /content
    python scripts/01_setup_much.py --root /content --skip-install
"""

from __future__ import annotations

import argparse
import py_compile
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/orailix/much.git"

# Everything src/ imports at module load time, minus the heavy ML stack that the
# runtime already provides with a matching CUDA build.
DEPS = [
    "python-decouple",
    "python-dotenv",
    "typer",
    "loguru",
    "pydantic>=2",
    "filelock",
    "tqdm",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "nltk",
    "diskcache",
    "wikipedia",
    "wptools",
    "joblib",
    "openai",
    "tiktoken",
    "much-segmenter",
    "datasets",
    "huggingface_hub",
    "sentence-transformers",
]

# (file, old text, replacement, skip_if_present).
# `skip_if_present` makes each patch idempotent: patch 2's replacement contains
# its own search text, so without the guard it would re-prepend on every run.
PATCHES: list[tuple[str, str, str, str]] = [
    (
        "src/annotation/human_evaluator.py",
        '{"\\n".join(textwrap.wrap(annotation.generation.output.strip()))}',
        "{wrapped_output}",
        "{wrapped_output}",
    ),
    (
        "src/annotation/human_evaluator.py",
        '    print(\n        f"""',
        '    wrapped_output = "\\n".join('
        "textwrap.wrap(annotation.generation.output.strip()))\n"
        '    print(\n        f"""',
        "wrapped_output = ",
    ),
    (
        "src/annotation/human_evaluator.py",
        """        all_labels.append(int(input(f"> {chunk_str.replace('\\n', ' ')}"""
        """    | Label=")))""",
        """        chunk_str_flat = chunk_str.replace("\\n", " ")\n"""
        """        all_labels.append(int(input(f"> {chunk_str_flat}    | Label=")))""",
        "chunk_str_flat = ",
    ),
]


def run(cmd: list[str], **kw) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, **kw)


def install() -> None:
    print("== installing MUCH's import-time dependencies ==")
    run([sys.executable, "-m", "pip", "install", "-q", *DEPS])


def clone(root: Path) -> Path:
    much = root / "much"
    if much.exists():
        print(f"== {much} already present ==")
    else:
        print("== cloning MUCH ==")
        run(["git", "clone", "--depth", "1", "-q", REPO, str(much)])
    return much


def patch(much: Path) -> None:
    """Idempotent. Only rewrites a file when the exact old text is still there."""
    print("== applying Python-3.11 compatibility patches ==")
    applied = 0
    for rel, old, new, skip_if in PATCHES:
        path = much / rel
        if not path.is_file():
            print(f"  skip (missing): {rel}")
            continue
        text = path.read_text()
        if skip_if in text:
            continue  # already patched
        if old in text:
            path.write_text(text.replace(old, new, 1))
            applied += 1
    print(f"  {applied} patch(es) applied" if applied else "  nothing to patch")


def verify(much: Path) -> bool:
    print("== compiling every source file ==")
    bad = []
    for f in sorted((much / "src").rglob("*.py")):
        try:
            py_compile.compile(str(f), doraise=True, cfile="/tmp/_grasp_check.pyc")
        except py_compile.PyCompileError as exc:
            bad.append((f, str(exc).strip().splitlines()[-1]))
    if bad:
        print(f"  {len(bad)} file(s) still fail to compile:")
        for f, msg in bad:
            print("   ", f, "|", msg)
        return False
    print("  all files compile")

    print("== importing the package ==")
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "import src\n"
        "from src.annotation import get_all_annotations\n"
        "from src.generation import (get_all_generations, template_question,\n"
        "                            get_model, get_tokenizer)\n"
        "print('  import OK on Python', sys.version.split()[0])\n" % str(much)
    )
    return run([sys.executable, "-c", code]) == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/content"))
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--import-data", action="store_true",
                    help="also run import-much and import-signals (needs HuggingFace)")
    args = ap.parse_args()

    if not args.skip_install:
        install()
    much = clone(args.root)
    patch(much)

    if not verify(much):
        print("\nSetup FAILED. Fix the errors above before continuing.")
        return 1

    if args.import_data:
        print("\n== downloading MUCH and the precomputed baseline signals ==")
        run([sys.executable, "-m", "src", "import-much"], cwd=str(much))
        run([sys.executable, "-m", "src", "import-signals"], cwd=str(much))
        out = much / "output"
        if out.exists():
            for child in sorted(out.iterdir()):
                print("  ", child.name)

    print("\nSetup complete.")
    print(f"  MUCH root : {much}")
    print("  next      : run the CCP reproduction checkpoint (notebook section 3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
