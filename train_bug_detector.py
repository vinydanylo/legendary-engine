import yaml
import random
import numpy as np
import torch
import time
from datetime import datetime

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from sklearn.metrics import precision_recall_fscore_support, matthews_corrcoef


def readConfig(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def buildDatasets(train_file: str, valid_file: str):
    data = load_dataset("json", data_files={
                        "train": train_file, "validation": valid_file})
    if "train" not in data or "validation" not in data:
        raise ValueError(
            "Could not load train/validation splits from the given files.")
    return data


def makeEncoder(tokenizer, max_len: int):
    def encode(batch):
        encoded = tokenizer(
            batch["code"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        encoded["labels"] = batch["label"]
        return encoded
    return encode


def computeMetrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    acc = float((preds == labels).mean())
    
    # Print detailed metrics
    print(f"\n[EVAL METRICS] Accuracy: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")
    print(f"[EVAL METRICS] F1: {f1:.4f}, MCC: {mcc:.4f}")
    
    return {"precision": p, "recall": r, "f1": f1, "mcc": mcc, "acc": acc}


class ProgressCallback(TrainerCallback):
    """Custom callback to show detailed training progress."""
    
    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[TRAINING START] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[CONFIG] Total epochs: {args.num_train_epochs}")
        print(f"[CONFIG] Batch size: {args.per_device_train_batch_size}")
        print(f"[CONFIG] Learning rate: {args.learning_rate}")
        print(f"[CONFIG] Total steps: {state.max_steps}")
        print(f"{'='*60}\n")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        print(f"\n{'='*50}")
        print(f"[EPOCH {state.epoch:.0f}/{args.num_train_epochs}] Starting...")
        print(f"{'='*50}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            elapsed = time.time() - self.start_time
            step = state.global_step
            total_steps = state.max_steps
            progress = step / total_steps * 100
            
            print(f"\n[STEP {step}/{total_steps} ({progress:.1f}%)] Loss: {logs['loss']:.4f}")
            print(f"[TIME] Elapsed: {elapsed/60:.1f}m, Step time: {elapsed/step:.2f}s/step")
            
            if 'learning_rate' in logs:
                print(f"[LR] {logs['learning_rate']:.2e}")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            print(f"\n[EPOCH {state.epoch:.0f}] Completed in {epoch_time/60:.2f} minutes")
    
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\n[EVALUATION] Running validation at step {state.global_step}...")
    
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print(f"[TRAINING COMPLETE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TOTAL TIME] {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
        print(f"[FINAL STEP] {state.global_step}/{state.max_steps}")
        print(f"{'='*60}\n")


class WeightedTrainer(Trainer):
    """A simple Trainer that supports class weights for CrossEntropyLoss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
        if class_weights is not None:
            # keep as tensor on the right device
            self.class_weights = torch.tensor(class_weights, dtype=torch.float)

    def computeLoss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    # 1) Read config
    cfg_path = "bug.yaml"
    cfg = readConfig(cfg_path)

    # 2) Device & seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BugTrain] Device: {device}")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3) Load data
    train_file = cfg["train_file"]
    valid_file = cfg["valid_file"]
    ds = buildDatasets(train_file, valid_file)

    # 4) Tokenizer & Model
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2).to(device)

    # 5) Tokenize
    max_len = int(cfg.get("max_seq_len", 512))
    encode = makeEncoder(tokenizer, max_len)
    print("[BugTrain] Tokenizing...")
    train_tok = ds["train"].map(
        encode, batched=True, remove_columns=["code", "label"])
    valid_tok = ds["validation"].map(
        encode, batched=True, remove_columns=["code", "label"])
    print("[BugTrain] Sizes:", len(train_tok), len(valid_tok))

    # 6) TrainingArgs
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        overwrite_output_dir=True,
        num_train_epochs=int(cfg.get("num_epochs", 4)),
        per_device_train_batch_size=int(cfg.get("batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("batch_size", 8)),
        learning_rate=float(cfg.get("lr", 2e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.06)),
        fp16=torch.cuda.is_available(),
        logging_steps=25,  # Log more frequently
        save_strategy="epoch",  # Match eval strategy
        save_total_limit=2,
        dataloader_num_workers=0,
        report_to="none",
        eval_strategy="epoch",
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )
    
    # Print training configuration
    print(f"\n[TRAINING CONFIG]")
    print(f"Model: {model_name}")
    print(f"Train samples: {len(train_tok)}")
    print(f"Valid samples: {len(valid_tok)}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {max_len}")
    print(f"Device: {device}")
    print(f"FP16: {args.fp16}")

    # 7) Create callback for progress monitoring
    progress_callback = ProgressCallback()
    
    # 8) Optional class weights
    class_weight = cfg.get("class_weight", None)
    if class_weight:
        print(f"\n[CONFIG] Using class weights: {class_weight}")
        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=valid_tok,
            processing_class=tokenizer,
            compute_metrics=computeMetrics,
            class_weights=class_weight,
            callbacks=[progress_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=valid_tok,
            processing_class=tokenizer,
            compute_metrics=computeMetrics,
            callbacks=[progress_callback],
        )

    # 9) Train & save
    print("\n[BugTrain] Starting training...")
    trainer.train()
    
    # Final evaluation
    print("\n[FINAL EVALUATION] Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"\n[FINAL RESULTS]")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    print(f"\n[SAVING MODEL] Saving to {cfg['output_dir']}")
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"[COMPLETE] Model and tokenizer saved successfully!")


if __name__ == "__main__":
    main()
