# ============================================================
# CELL 1 — IMPORTS & CONFIG
# ============================================================
#%%
import os, random, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import librosa
import whisper

warnings.filterwarnings("ignore")

TRAIN_AUDIO_DIR = r"C:\Anand\shl\shl-audio-scoring-challenge\dataset\audios\train"
TEST_AUDIO_DIR  = r"C:\Anand\shl\shl-audio-scoring-challenge\dataset\audios\test"
TRAIN_CSV       = r"C:\Anand\shl\shl-audio-scoring-challenge\dataset\csvs\train.csv"
TEST_CSV        = r"C:\Anand\shl\shl-audio-scoring-challenge\dataset\csvs\test.csv"
TRAIN_OUT       = "outputs/train_transcripts.csv"
TEST_OUT        = "outputs/test_transcripts.csv"
CKPT_PATH       = "outputs/epoch10_model.pt"
SUBMISSION_PATH = "outputs/submission.csv"
MODEL_NAME      = "microsoft/deberta-v3-base"
WHISPER_MODEL   = "small"
MAX_LEN         = 128
DROPOUT         = 0.2
VAL_SIZE        = 0.2
EPOCHS          = 10
BATCH_SIZE      = 8
LR              = 1e-5
WEIGHT_DECAY    = 1e-2
WARMUP_RATIO    = 0.2
GRAD_CLIP       = 1.0
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FP16            = torch.cuda.is_available()
PIN_MEMORY      = torch.cuda.is_available()

os.makedirs("outputs", exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed()
print(f"Device: {DEVICE}  |  FP16: {FP16}")


# ============================================================
# CELL 2 — TRANSCRIBE AND SAVE CSVs (uses librosa, no ffmpeg)
# Run once. Re-running will reload saved CSVs.
# ============================================================

# ── Load CSVs ─────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
train_df.columns = [c.strip().lower() for c in train_df.columns]
test_df.columns  = [c.strip().lower() for c in test_df.columns]

fn_col    = next(c for c in train_df.columns if any(k in c for k in ("file","audio","name")))
label_col = next(c for c in train_df.columns if c != fn_col)
test_fn   = next(c for c in test_df.columns  if any(k in c for k in ("file","audio","name")))

train_files  = [str(f) for f in train_df[fn_col].tolist()]
test_files   = [str(f) for f in test_df[test_fn].tolist()]
train_labels = train_df[label_col].tolist()

print(f"Train: {len(train_files)} files  e.g. {train_files[:3]}")
print(f"Test : {len(test_files)}  files  e.g. {test_files[:3]}")

# ── Load Whisper ──────────────────────────────────────────
print(f"\nLoading Whisper '{WHISPER_MODEL}' ...")
asr = whisper.load_model(WHISPER_MODEL)

def transcribe_with_librosa(fn: str, audio_dir: str) -> str:
    """
    Load audio using librosa (no ffmpeg needed) and pass
    the numpy array directly to Whisper for transcription.
    """
    fpath = os.path.join(audio_dir, fn + ".wav")
    if not os.path.exists(fpath):
        print(f"  [MISSING] {fpath}")
        return ""
    try:
        # librosa loads audio as float32 numpy array at 16kHz (Whisper's required rate)
        audio, _ = librosa.load(fpath, sr=16000, mono=True)
        # Pass numpy array directly to Whisper — bypasses ffmpeg entirely
        result = asr.transcribe(
            audio,
            language="en",
            fp16=torch.cuda.is_available(),
            verbose=False,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        return result["text"].strip()
    except Exception as e:
        print(f"  [ERROR] {fn}: {e}")
        return ""

# ── Transcribe train ──────────────────────────────────────
train_transcripts = []
print(f"\nTranscribing {len(train_files)} train files ...")
for fn in tqdm(train_files):
    train_transcripts.append(transcribe_with_librosa(fn, TRAIN_AUDIO_DIR))

# ── Transcribe test ───────────────────────────────────────
test_transcripts = []
print(f"\nTranscribing {len(test_files)} test files ...")
for fn in tqdm(test_files):
    test_transcripts.append(transcribe_with_librosa(fn, TEST_AUDIO_DIR))

# ── Save CSVs ─────────────────────────────────────────────
pd.DataFrame({
    "filename"  : train_files,
    "label"     : train_labels,
    "transcript": train_transcripts,
}).to_csv(TRAIN_OUT, index=False)

pd.DataFrame({
    "filename"  : test_files,
    "transcript": test_transcripts,
}).to_csv(TEST_OUT, index=False)

empty_tr = sum(1 for t in train_transcripts if not t)
empty_te = sum(1 for t in test_transcripts  if not t)
print(f"\nSaved → {TRAIN_OUT}  (empty: {empty_tr}/{len(train_files)})")
print(f"Saved → {TEST_OUT}   (empty: {empty_te}/{len(test_files)})")
print("\nSample train transcripts:")
for fn, lbl, txt in zip(train_files[:5], train_labels[:5], train_transcripts[:5]):
    print(f"  [{lbl}] {fn}: {txt[:80]}")
print("\nSample test transcripts:")
for fn, txt in zip(test_files[:5], test_transcripts[:5]):
    print(f"  {fn}: {txt[:80]}")


# ============================================================
# CELL 3 — DATASET, MODEL, HELPERS
# ============================================================

class GrammarDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None):
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


class DeBERTaRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        hidden        = self.backbone.config.hidden_size
        self.drop     = nn.Dropout(DROPOUT)
        self.head     = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        try:
            out = self.backbone(**kwargs, token_type_ids=token_type_ids)
        except TypeError:
            out = self.backbone(**kwargs)
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.head(self.drop(pooled)).squeeze(-1)


def pearson_loss(preds, labels):
    px  = preds  - preds.mean()
    py  = labels - labels.mean()
    cor = (px * py).sum() / (px.norm() * py.norm() + 1e-8)
    return 1.0 - cor

def combined_loss(preds, labels):
    return nn.MSELoss()(preds, labels) + 0.5 * pearson_loss(preds, labels)

def to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def fwd(model, batch):
    return model(
        input_ids      = batch["input_ids"],
        attention_mask = batch["attention_mask"],
        token_type_ids = batch.get("token_type_ids"),
    )

def train_epoch(model, loader, optimizer, scheduler, scaler):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        batch = to_device(batch, DEVICE)
        optimizer.zero_grad()
        with autocast(enabled=FP16):
            loss = combined_loss(fwd(model, batch), batch["labels"])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    preds, labels = [], []
    for batch in tqdm(loader, desc="  val  ", leave=False):
        batch = to_device(batch, DEVICE)
        with autocast(enabled=FP16):
            p = fwd(model, batch)
        preds.extend(p.cpu().float().numpy())
        labels.extend(batch["labels"].cpu().numpy())
    return float(np.sqrt(mean_squared_error(labels, preds)))

@torch.no_grad()
def run_predict(model, loader):
    model.eval()
    preds = []
    for batch in tqdm(loader, desc="  predict", leave=False):
        batch = to_device(batch, DEVICE)
        with autocast(enabled=FP16):
            p = fwd(model, batch)
        preds.extend(p.cpu().float().numpy())
    return np.array(preds)

print("Dataset, model and helpers ready ✔")


# ============================================================
# CELL 4 — TRAIN AND PREDICT
# ============================================================

set_seed()

# ── Load transcript CSVs ──────────────────────────────────
tr_df = pd.read_csv(TRAIN_OUT)
te_df = pd.read_csv(TEST_OUT)
tr_df["transcript"] = tr_df["transcript"].fillna("").astype(str)
te_df["transcript"] = te_df["transcript"].fillna("").astype(str)

train_texts  = tr_df["transcript"].tolist()
train_labels = tr_df["label"].astype(float).tolist()
test_texts   = te_df["transcript"].tolist()
test_files   = te_df["filename"].tolist()

empty_tr = sum(1 for t in train_texts if not t.strip())
empty_te = sum(1 for t in test_texts  if not t.strip())
print(f"Empty transcripts — train: {empty_tr}  test: {empty_te}")

print("\nSample train transcripts:")
for i in range(min(3, len(train_texts))):
    print(f"  [{train_labels[i]:.1f}] {train_texts[i][:100]}")
print("\nSample test transcripts:")
for t in test_texts[:3]:
    print(f"  {t[:100]}")

# ── Train / Val split ─────────────────────────────────────
tr_texts, val_texts, tr_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=VAL_SIZE, random_state=SEED
)
print(f"\nSplit — Train: {len(tr_texts)}  Val: {len(val_texts)}  Test: {len(test_texts)}")

# ── Tokenizer ─────────────────────────────────────────────
print(f"\nLoading tokenizer: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Datasets & Loaders ────────────────────────────────────
tr_ds   = GrammarDataset(tr_texts,   tr_labels,  tokenizer)
val_ds  = GrammarDataset(val_texts,  val_labels, tokenizer)
test_ds = GrammarDataset(test_texts, None,       tokenizer)

tr_loader   = DataLoader(tr_ds,   batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=PIN_MEMORY)
val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=PIN_MEMORY)

# ── Model ─────────────────────────────────────────────────
model = DeBERTaRegressor().to(DEVICE)
print(f"Model on {DEVICE}  |  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ── Optimizer & Scheduler ─────────────────────────────────
optimizer = AdamW(
    [
        {"params": model.backbone.parameters(), "lr": LR},
        {"params": model.head.parameters(),     "lr": LR * 10},
    ],
    weight_decay=WEIGHT_DECAY,
)
total_steps  = len(tr_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
scaler       = GradScaler(enabled=FP16)

# ── Training loop ─────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Training — {EPOCHS} epochs  |  Loss: MSE + Pearson")
print(f"{'='*60}")

val_rmse = None
for epoch in range(1, EPOCHS + 1):
    tr_loss  = train_epoch(model, tr_loader, optimizer, scheduler, scaler)
    val_rmse = eval_epoch(model, val_loader)
    print(f"  Epoch {epoch:02d}/{EPOCHS}  |  train_loss={tr_loss:.4f}  val_RMSE={val_rmse:.4f}")

# ── Save epoch 10 checkpoint ──────────────────────────────
torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, CKPT_PATH)
print(f"\nEpoch 10 checkpoint saved → {CKPT_PATH}")

# ── Predict on test (model already in epoch 10 state) ─────
print("Running test predictions ...")
test_preds = np.clip(run_predict(model, test_loader), 0.0, 5.0)

# ── Save submission ───────────────────────────────────────
submission = pd.DataFrame({"filename": test_files, "label": test_preds})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n{'='*60}")
print(f"  Final Val RMSE   : {val_rmse:.4f}")
print(f"  Submission saved : {SUBMISSION_PATH}")
print(f"{'='*60}")
print(submission.head(10).to_string(index=False))
# %%
