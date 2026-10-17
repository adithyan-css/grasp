#!/usr/bin/env python3
"""
Build kaggle/grasp-full-run/GRASP_kaggle.ipynb: a self-contained, run-to-the-end
Kaggle notebook. The project code (src/ and scripts/) is embedded as a base64 zip,
so no Drive mount or dataset upload is needed. Re-run this after any code change.

The notebook needs, on Kaggle: GPU accelerator, Internet on, and a secret named
HF_TOKEN (Add-ons -> Secrets) whose account is approved for the Llama/Gemma/
Ministral models MUCH uses.

    python kaggle/build_notebook.py
"""

from __future__ import annotations

import base64
import io
import json
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "kaggle" / "grasp-full-run"


def code_zip_b64() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("src", "scripts"):
            for f in sorted((PROJECT / sub).rglob("*.py")):
                if "__pycache__" in f.parts:
                    continue
                z.write(f, f.relative_to(PROJECT).as_posix())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n"),
    }


CELLS = [
    md("""
# GRASP v2 — multi-head extraction + measured overhead (all 8 configs)

Self-contained: the project code is embedded below. Needs **GPU (2x T4)** and
**Internet on**. HuggingFace login uses the `HF_TOKEN` secret if attached, otherwise
a device code (open a URL, enter the code).

This run only produces features and timings. It extracts the top-4 variance
heads per layer (`--head-select variance_topk --top-k 4`) for every MUCH generation
config, builds the claim table, and measures GRASP's runtime overhead against real
generation on the same GPU. It does **not** score the test split: development and
the single v2 test evaluation run locally from the downloaded table, with selection
rules fixed beforehand.
"""),
    code("""
import base64, io, json, os, subprocess, sys, time, zipfile
from pathlib import Path

WORK    = Path('/kaggle/working')
PROJECT = WORK / 'ai_project'
RESULTS = WORK / 'results'
MUCH_PARENT = Path('/kaggle/tmp')          # big data stays out of the saved output
MUCH    = MUCH_PARENT / 'much'
for p in (PROJECT, RESULTS, MUCH_PARENT):
    p.mkdir(parents=True, exist_ok=True)

CODE_ZIP = '__CODE_ZIP__'
zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_ZIP))).extractall(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))

LOG = RESULTS / 'run_log.txt'

def sh(cmd, cwd=None, check=True):
    \"\"\"Run a command, stream its output to the cell and to run_log.txt.\"\"\"
    print('$', cmd, flush=True)
    t0 = time.time()
    with open(LOG, 'a', encoding='utf-8') as log:
        log.write(f'\\n$ {cmd}\\n')
        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
        rc = proc.wait()
        log.write(f'[exit {rc}, {time.time() - t0:.0f}s]\\n')
    if check and rc != 0:
        raise RuntimeError(f'step failed (exit {rc}): {cmd}')
    return rc

def disk(tag=''):
    sh(f'df -h {WORK} {MUCH_PARENT} / | sed "1!s/^/  /"; echo "  [{tag}]"; du -sh {RESULTS} {MUCH_PARENT} ~/.cache/huggingface 2>/dev/null', check=False)

sh('nvidia-smi -L')
print('python', sys.version.split()[0])
disk('start')
"""),
    code("""
# HuggingFace login. Uses the HF_TOKEN Kaggle secret if one is attached; otherwise
# falls back to HuggingFace's device-code login (open a URL, enter a short code),
# the same flow Colab's login() shows. Implemented with plain requests so it does
# not depend on the preinstalled huggingface_hub version. The token is never printed.
import requests

def device_code_login(timeout_s=1800):
    hf, client = 'https://huggingface.co', '26be6b09-91c5-47da-9861-d2d2bb7a7e36'
    info = requests.post(f'{hf}/oauth/device', data={'client_id': client}, timeout=30)
    info.raise_for_status(); info = info.json()
    url = info.get('verification_uri_complete') or info['verification_uri']
    print('=' * 60)
    print(f"HF-DEVICE-LOGIN url={url} code={info['user_code']}")
    print('Open the URL, sign in as your HF user, enter the code, authorize.')
    print('=' * 60, flush=True)
    interval = int(info.get('interval', 5))
    deadline = time.time() + min(int(info.get('expires_in', 900)), timeout_s)
    while time.time() < deadline:
        time.sleep(interval)
        try:
            r = requests.post(f'{hf}/oauth/token', timeout=30, data={
                'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                'device_code': info['device_code'], 'client_id': client}).json()
        except Exception:
            continue
        if 'access_token' in r:
            print('authorized; token valid for', r.get('expires_in', '?'), 's', flush=True)
            return r['access_token']
        err = r.get('error')
        if err == 'slow_down':
            interval += 5
        elif err == 'expired_token':
            return None                      # caller asks for a fresh code
        elif err not in (None, 'authorization_pending'):
            raise SystemExit(f'HF device login failed: {err}')
    return None

token = None
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret('HF_TOKEN')
    print('using the HF_TOKEN Kaggle secret')
except Exception:
    print('no HF_TOKEN secret attached; using device-code login')
t_end = time.time() + 2 * 3600            # keep offering fresh codes for 2 hours
while not token and time.time() < t_end:
    token = device_code_login()
    if not token:
        print('code expired without authorization; requesting a new one', flush=True)
if not token:
    raise SystemExit('HF device login was never authorized.')
os.environ['HF_TOKEN'] = token          # inherited by every subprocess below
tok_file = Path.home() / '.cache' / 'huggingface' / 'token'
tok_file.parent.mkdir(parents=True, exist_ok=True)
tok_file.write_text(token)
who = requests.get('https://huggingface.co/api/whoami-v2',
                   headers={'Authorization': f'Bearer {token}'}, timeout=30).json()
print('HF login ok as', who.get('name'))
"""),
    md("## 1. MUCH setup, data, protocol checks"),
    code("""
sh(f'{sys.executable} {PROJECT}/scripts/01_setup_much.py --root {MUCH_PARENT}')
sh(f'{sys.executable} -m src import-much', cwd=MUCH)
sh(f'{sys.executable} -m src import-signals', cwd=MUCH)
sh(f'{sys.executable} {PROJECT}/src/tests/test_grasp.py')
sh(f'{sys.executable} {PROJECT}/src/tests/test_much_eval.py --much-root {MUCH}')
disk('after MUCH data')
"""),
    code("""
sys.path.insert(0, str(MUCH))
from collections import Counter
from src.annotation import get_all_annotations
from src.generation import get_all_generations
from huggingface_hub import model_info

anns = get_all_annotations(); gens = get_all_generations()
print(f'{len(anns)} annotations, {len(gens)} generations')
human = [a for a in anns.values() if any(not k.startswith('gpt-') for k in a.labels)]
print('human-annotated (test split):', len(human))
for m in sorted({g.generation_cfg.model_name for g in gens.values()}):
    try:
        model_info(m); print('  OK      ', m)
    except Exception as exc:
        print('  BLOCKED ', m, type(exc).__name__)
"""),
    md("## 2. CCP checkpoint (go/no-go)"),
    code("""
sh(f'{sys.executable} {PROJECT}/scripts/04_reproduce_ccp.py --much-root {MUCH} '
   f'--signal ccp_24_8 --out {RESULTS}/ccp_reproduction.json')
"""),
    md("## 3. Attention extraction (GPU) — all 8 generation configs, corrected profile"),
    code("""
import torch
n_gpu = torch.cuda.device_count()
cfgs = {}
for g in gens.values():
    c = g.generation_cfg
    cfgs.setdefault(g.generation_config, (c.model_name, c.temperature))
# small models first, so most of the table exists even if an 8B run fails
order = sorted(cfgs, key=lambda k: ('8B' in cfgs[k][0], cfgs[k][0], cfgs[k][1]))
for k in order:
    print(k, cfgs[k])

CACHES, STATUS = [], {}
for cfg in order:
    model, temp = cfgs[cfg]
    out = RESULTS / f'cache_k4_{cfg}'
    shard = '--device-map auto' if ('8B' in model and n_gpu > 1) else ''
    t0 = time.time()
    rc = sh(f'{sys.executable} {PROJECT}/scripts/02_extract_attention.py --much-root {MUCH} '
            f'--config {cfg} --out {out} --calib 200 --profile corrected '
            f'--head-select variance_topk --top-k 4 {shard}', check=False)
    n = len(list((out / 'features').glob('*.npy'))) if (out / 'features').exists() else 0
    STATUS[cfg] = {'model': model, 'temp': temp, 'exit': rc, 'features': n,
                   'minutes': round((time.time() - t0) / 60, 1)}
    print('STATUS', cfg, STATUS[cfg], flush=True)
    if rc == 0 and (out / 'normaliser.npz').exists():
        CACHES.append(out)
    import gc; gc.collect(); torch.cuda.empty_cache()
(RESULTS / 'extraction_status.json').write_text(json.dumps(STATUS, indent=2))
disk('after extraction')
assert CACHES, 'no config extracted'
"""),
    md("## 4. Claim table (all splits; no scoring here)"),
    code("""
cache_args = ' '.join(f'--cache {c}' for c in CACHES)
sh(f'{sys.executable} {PROJECT}/scripts/08_claim_table.py --much-root {MUCH} {cache_args} '
   f'--tau 16.0 --alpha 0.9 --out {RESULTS}/claims_k4.npz')
"""),
    md("## 5. Measured runtime overhead vs real generation (same GPU), one config per model"),
    code("""
done_models = set()
for cfg in order:
    model, temp = cfgs[cfg]
    if model in done_models or abs(temp - 0.7) > 1e-6:
        continue
    cache = RESULTS / f'cache_k4_{cfg}'
    if not (cache / 'selected_heads.json').exists():
        continue
    shard = '--device-map auto' if ('8B' in model and n_gpu > 1) else ''
    sh(f'{sys.executable} {PROJECT}/scripts/11_overhead.py --much-root {MUCH} --config {cfg} '
       f'--heads-file {cache}/selected_heads.json --n 20 {shard} '
       f'--out {RESULTS}/overhead_{cfg}.json', check=False)
    done_models.add(model)
    import gc; gc.collect(); torch.cuda.empty_cache()
"""),
    code("""
# One archive with everything except the bulky per-generation feature files.
import shutil
for f in sorted(RESULTS.glob('*')):
    print(f'{f.stat().st_size if f.is_file() else 0:>10}  {f.name}')
shutil.make_archive(str(WORK / 'grasp_results'), 'zip', RESULTS)
print('done')
"""),
]


def main() -> None:
    blob = code_zip_b64()
    cells = [dict(c) for c in CELLS]
    for c in cells:
        if "__CODE_ZIP__" in c["source"]:
            c["source"] = c["source"].replace("__CODE_ZIP__", blob)
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "GRASP_kaggle.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, code zip {len(blob) // 1024} KB)")


if __name__ == "__main__":
    main()
