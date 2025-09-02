import re
import json
import argparse
from pathlib import Path
from collections import Counter
from typing import Iterable

SAMPLE_CHARS = 4000

CSHARP_PATTERNS = [
    r"\busing\s+[A-Z][A-Za-z0-9_.]+;",
    r"\bnamespace\s+[A-Z][A-Za-z0-9_.]+\b",
    r"\bpublic\s+(class|interface|enum|struct)\b",
    r"\bclass\s+[A-Z][A-Za-z0-9_]*\b",
    r"\bprivate\s+|protected\s+|internal\s+",
    r"\bConsole\.(WriteLine|Write)\s*\(",
    r"\bnew\s+[A-Z][A-Za-z0-9_]*\s*\(",
    r"\bbool\b|\bint\b|\bstring\b|\bvar\b|\bList<",
    r"\b[A-Z][A-Za-z0-9_]*<[A-Za-z0-9_,\s?]+>\s+[A-Za-z_][A-Za-z0-9_]*",
    r"\btry\s*\{[\s\S]*?\}\s*catch\s*\(",
    r"\bget;\s*set;\b",
    r";\s*$",
]

PY_HARD = [
    r"^\s*def\s+\w+\(",
    r"^\s*class\s+\w+\s*:\s*$",
    r"^\s*import\s+\w+\s*$",
    r"^\s*from\s+\w+\s+import\s+\w+",
    r":\s*$",
    r"^\s*return\s+",
    r"'''|\"\"\"",
]

JS_HARD = [
    r"\bfunction\s+\w+\(",
    r"\b=>\s*\{",
    r"\bconsole\.log\(",
    r"\bexport\s+(const|function|class)\b",
    r"\bmodule\.exports\b",
    r"\brequire\(",
]

JAVA_HARD = [
    r"\bpackage\s+[A-Za-z0-9_.]+;",
    r"\bimport\s+java\.",
    r"\bSystem\.out\.",
    r"\bpublic\s+static\s+void\s+main\s*\(",
    r"\b@[A-Z]\w+",
    r"\bboolean\b|\bchar\b|\bInteger\b|\bLong\b",
]


def compileData(patterns: Iterable[str]):
    return [re.compile(p, re.MULTILINE) for p in patterns]


CSHARP_RX = compileData(CSHARP_PATTERNS)
PY_RX = compileData(PY_HARD)
JS_RX = compileData(JS_HARD)
JAVA_RX = compileData(JAVA_HARD)


def score(text, regexes):
    sample = text[:SAMPLE_CHARS]
    return sum(1 for rx in regexes if rx.search(sample))


def csharpIdentifier(code, min_csharp_score=2, min_margin=1, others_cap=2):
    sample = code[:SAMPLE_CHARS]
    py = score(sample, PY_RX)
    js = score(sample, JS_RX)
    java = score(sample, JAVA_RX)
    others_max = max(py, js, java)
    if others_max > others_cap:
        return False, "others_high"
    cs = score(sample, CSHARP_RX)
    if cs < min_csharp_score:
        return False, "cs_low"
    if (cs - others_max) < min_margin:
        return False, "low_margin"
    return True, "ok"


def filterFile(
    in_path,
    out_path,
    min_csharp_score=2,
    min_margin=1,
    others_cap=2,
    progress_every=50_000,
    max_keep=0,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    reasons = Counter()

    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                reasons["bad_json"] += 1
                continue

            code = obj.get("code", "")
            try:
                label = int(obj.get("label", 0))
            except Exception:
                label = 0
                reasons["bad_label"] += 1

            ok, why = csharpIdentifier(
                code,
                min_csharp_score=min_csharp_score,
                min_margin=min_margin,
                others_cap=others_cap,
            )

            if not ok:
                reasons[why] += 1
            else:
                fout.write(json.dumps(
                    {"code": code, "label": label, "lang": "csharp"}, ensure_ascii=False) + "\n")
                kept += 1
                if max_keep and kept >= max_keep:
                    break

            if progress_every and (total % progress_every == 0):
                print(f"[{in_path}] processed={total:,} kept={kept:,}", flush=True)

    return total, kept, reasons


def labelCounts(path):
    c = Counter()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    c[int(json.loads(line)["label"])] += 1
                except Exception:
                    continue
    except FileNotFoundError:
        return Counter()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_train",  required=True)
    ap.add_argument("--in_valid",  required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_valid", required=True)

    ap.add_argument("--min_csharp_score", type=int, default=2)
    ap.add_argument("--min_margin",       type=int, default=1)
    ap.add_argument("--others_cap",       type=int, default=2)
    ap.add_argument("--progress_every", type=int, default=50_000)
    ap.add_argument("--max_keep",       type=int, default=0)

    args = ap.parse_args()

    t_total, t_kept, t_reasons = filterFile(
        args.in_train, args.out_train,
        min_csharp_score=args.min_csharp_score,
        min_margin=args.min_margin,
        others_cap=args.others_cap,
        progress_every=args.progress_every,
        max_keep=args.max_keep,
    )
    v_total, v_kept, v_reasons = filterFile(
        args.in_valid, args.out_valid,
        min_csharp_score=args.min_csharp_score,
        min_margin=args.min_margin,
        others_cap=args.others_cap,
        progress_every=args.progress_every,
        max_keep=args.max_keep,
    )


if __name__ == "__main__":
    main()
