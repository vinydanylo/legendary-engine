import argparse
import json
import sys
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_CFG = Path("configs/bug_java.yaml")


def sanitize_code(code_text: str) -> str:
    """
    Sanitize code by removing comments and all line breaks/whitespace.
    Handles C#, Java, and other C-style languages.
    """
    # Remove single-line comments (// ...)
    code_text = re.sub(r'//.*?(?=\n|$)', '', code_text)

    # Remove multi-line comments (/* ... */)
    code_text = re.sub(r'/\*.*?\*/', '', code_text, flags=re.DOTALL)

    # Remove XML/HTML comments (<!-- ... -->)
    code_text = re.sub(r'<!--.*?-->', '', code_text, flags=re.DOTALL)

    # Remove all line breaks and normalize whitespace
    code_text = re.sub(r'\r\n|\r|\n', ' ', code_text)
    
    # Replace multiple whitespace characters with single spaces
    code_text = re.sub(r'\s+', ' ', code_text)

    # Strip leading and trailing whitespace
    code_text = code_text.strip()

    return code_text


def load_yaml_safe(path: Path):
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_threshold_from_file(model_dir: Path, default_value: float = 0.5) -> float:
    tfile = model_dir
    if tfile.exists():
        try:
            obj = json.loads(tfile.read_text(encoding="utf-8"))
            t = float(obj.get("threshold", default_value))
            if t < 0.0:
                t = 0.0
            if t > 1.0:
                t = 1.0
            return t
        except Exception:
            pass
    return default_value


class BugInfer:
    def __init__(self, model_dir, max_len=512, threshold=None, device=None):
        self.model_dir = Path(model_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.max_len = int(max_len)
        if threshold is None:
            self.threshold = load_threshold_from_file(self.model_dir, 0.5)
        else:
            self.threshold = float(threshold)

    def predict_one(self, code_text: str):
        sanitized_code = sanitize_code(code_text)
        with torch.no_grad():
            enc = self.tokenizer(
                sanitized_code,
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            p_bug = float(probs[1].detach().cpu().item())
            label = 1 if p_bug >= self.threshold else 0
            return {"p_bug": p_bug, "label": label}

    def predict_many(self, code_list, batch_size=32):
        sanitized_code_list = [sanitize_code(code) for code in code_list]
        results = []
        with torch.no_grad():
            for start in range(0, len(sanitized_code_list), batch_size):
                chunk = sanitized_code_list[start:start + batch_size]
                enc = self.tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(
                    logits, dim=-1)[:, 1].detach().cpu().tolist()
                for p in probs:
                    results.append(
                        {"p_bug": float(p), "label": 1 if p >= self.threshold else 0})
        return results


def read_text_input(path: Path, split_lines=False):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    if split_lines:
        return [ln for ln in (l.strip() for l in txt.splitlines()) if ln]
    return [txt]


def read_jsonl_codes(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                code = obj["code"]
                label = obj.get("label")
                out.append({"code": code, "label": label})
            except Exception:
                # skip bad line
                continue
    return out


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Simple bug detector inference")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--code", type=str, default=None,
                        help="Single code snippet")
    parser.add_argument("--in_file", type=str, default=None,
                        help="Path to .java/.txt file")
    parser.add_argument("--split_lines", action="store_true",
                        help="For --in_file: treat each non-empty line as a snippet")
    parser.add_argument("--jsonl_in", type=str, default=None,
                        help="JSONL with objects that contain a 'code' field")

    parser.add_argument("--jsonl_out", type=str,
                        default=None, help="Write results to JSONL")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Write results to CSV (idx,p_bug,label)")

    args = parser.parse_args()

    cfg = load_yaml_safe(DEFAULT_CFG) if DEFAULT_CFG.exists() else {}

    model_dir = Path(args.model_dir or cfg.get(
        "output_dir", "codebert_bug"))
    max_len = int(args.max_len or cfg.get("max_seq_len", 512))
    threshold = args.threshold

    chosen = 0
    if args.code is not None:
        chosen += 1
    if args.in_file is not None:
        chosen += 1
    if args.jsonl_in is not None:
        chosen += 1
    if chosen != 1:
        print(
            "Please pass exactly one of: --code OR --in_file OR --jsonl_in", file=sys.stderr)
        sys.exit(2)

    infer = BugInfer(model_dir=model_dir, max_len=max_len, threshold=threshold)

    if args.code is not None:
        res = infer.predict_one(args.code)
        out = {"p_bug": round(
            res["p_bug"], 6), "label": res["label"], "threshold": infer.threshold}
        print(json.dumps(out, ensure_ascii=False))
        return

    if args.in_file is not None:
        p = Path(args.in_file)
        snippets = read_text_input(p, split_lines=args.split_lines)
        results = infer.predict_many(snippets, batch_size=args.batch_size)

        for i, r in enumerate(results, 1):
            print(json.dumps({"idx": i, "p_bug": round(
                r["p_bug"], 6), "label": r["label"]}, ensure_ascii=False))

        if args.csv_out:
            csvp = Path(args.csv_out)
            csvp.parent.mkdir(parents=True, exist_ok=True)
            with csvp.open("w", encoding="utf-8") as f:
                f.write("idx,p_bug,label\n")
                for i, r in enumerate(results, 1):
                    f.write(f"{i},{r['p_bug']:.6f},{r['label']}\n")
        return

    if args.jsonl_in is not None:
        inp = Path(args.jsonl_in)
        rows = read_jsonl_codes(inp)
        codes = [r["code"] for r in rows]
        preds = infer.predict_many(codes, batch_size=args.batch_size)

        for r, pr in zip(rows, preds):
            r["p_bug"] = float(pr["p_bug"])
            r["pred"] = int(pr["label"])
            r["threshold"] = infer.threshold

        preview = rows[:5]
        for r in preview:
            print(json.dumps(r, ensure_ascii=False))
        if len(rows) > 5:
            print(f"... ({len(rows)-5} more)")

        if args.jsonl_out:
            outp = Path(args.jsonl_out)
            write_jsonl(outp, rows)


if __name__ == "__main__":
    main()
