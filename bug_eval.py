import os
import yaml
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


def computeMetrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.astype(int)

    # stability: subtract per-row max
    row_max = logits.max(axis=1, keepdims=True)
    stable = logits - row_max

    # exponentiate
    exp_scores = np.exp(stable)

    # normalize to probabilities
    sums = exp_scores.sum(axis=1, keepdims=True)
    probs = exp_scores / sums

    # Probability of "bug" class (index 1)
    p_bug = probs[:, 1]

    # Argmax at threshold 0.5
    preds = (p_bug >= 0.5).astype(int)

    # Confusion matrix parts
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())

    # Metrics at 0.5 threshold
    total = len(labels)
    acc = float((preds == labels).mean())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    # MCC (argmax)
    denom = max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt(denom)

    # Best F1 by sweeping threshold
    best_f1 = 0.0
    best_t = 0.5
    best_p = 0.0
    best_r = 0.0

    thresholds = np.linspace(0.1, 0.9, 81)
    for t in thresholds:
        pred_t = (p_bug >= t).astype(int)
        tp_t = int(((pred_t == 1) & (labels == 1)).sum())
        fp_t = int(((pred_t == 1) & (labels == 0)).sum())
        fn_t = int(((pred_t == 0) & (labels == 1)).sum())
        p_t = tp_t / max(tp_t + fp_t, 1)
        r_t = tp_t / max(tp_t + fn_t, 1)
        if (p_t + r_t) == 0:
            f1_t = 0.0
        else:
            f1_t = 2 * p_t * r_t / (p_t + r_t)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_t = float(t)
            best_p = p_t
            best_r = r_t

    return {
        "acc": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "best_f1": round(float(best_f1), 4),
        "best_thresh": round(float(best_t), 3),
        "best_precision": round(float(best_p), 4),
        "best_recall": round(float(best_r), 4),
        "mcc": round(float(mcc), 4),
        "total_examples": int(total),
        "tp_fp_fn_tn": [tp, fp, fn, tn],
    }


def main():
    # 1) Read config
    cfg_path = "configs/bug_java.yaml"
    if not os.path.exists(cfg_path):
        print("Config file not found:", cfg_path)
        return
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2) Where to load model and validation data from
    model_dir = cfg["output_dir"]
    valid_path = cfg["valid_file"]
    max_len = int(cfg.get("max_seq_len", 512))
    batch_size = int(cfg.get("batch_size", 8))

    print("[Eval] Model dir:", model_dir)
    print("[Eval] Valid file:", valid_path)

    # 3) Load tokenizer and model
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # 4) Load validation set (expects JSONL with fields: code, label)
    data = load_dataset("json", data_files={"validation": valid_path})

    # 5) Tokenize function
    def encode(batch):
        out = tok(
            batch["code"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        out["labels"] = batch["label"]
        return out

    print("[Eval] Tokenizing validation set...")
    valid = data["validation"].map(
        encode, batched=True, remove_columns=["code", "label"])
    print("[Eval] Validation size:", len(valid))

    # 6) Evaluation settings
    args = TrainingArguments(
        output_dir="eval_out",
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
        report_to="none",
        evaluation_strategy="no",
    )

    # 7) Run evaluation
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tok,
        compute_metrics=computeMetrics,
    )
    metrics = trainer.evaluate(eval_dataset=valid)

    # 8) Print results clearly
    for key in [
        "acc", "precision", "recall", "f1",
        "best_f1", "best_thresh", "best_precision", "best_recall",
        "mcc", "total_examples"
    ]:
        if key in metrics:
            print(f"{key}: {metrics[key]}")
    if "tp_fp_fn_tn" in metrics:
        tp, fp, fn, tn = metrics["tp_fp_fn_tn"]
        print(f"TP/FP/FN/TN: {tp}/{fp}/{fn}/{tn}")


if __name__ == "__main__":
    main()
