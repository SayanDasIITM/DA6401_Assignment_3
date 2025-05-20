#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Seq2Seq Transliteration with WandB Sweep (GPU‐ready, DataParallel)

Includes:
- Scheduled‐sampling teacher‐forcing decay
- Meaningful WandB run names (hyperparams)
- Both character‐level and word‐level accuracy printed and logged each epoch
- Enhanced W&B table visualization for predictions
- Test predictions saved to file for GitHub upload
"""

import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

# -----------------------------------
# 1. Argument Parsing & Environment
# -----------------------------------

parser = argparse.ArgumentParser(description="Seq2Seq Transliteration with WandB Sweep")
parser.add_argument("--train_path", type=str,
                   default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.train.tsv")
parser.add_argument("--dev_path", type=str,
                   default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.dev.tsv")
parser.add_argument("--test_path", type=str,
                   default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.test.tsv")
parser.add_argument("--project", type=str, default="DA6401_Assignment3")
parser.add_argument("--predictions_dir", type=str, default="predictions_vanilla")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"==> Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("!! Warning: CUDA not available. Running on CPU.")

# Create predictions directory
os.makedirs(args.predictions_dir, exist_ok=True)

wandb.login()

# -----------------------------------
# 2. Data Utilities
# -----------------------------------

def build_vocab(paths, min_freq=1):
    freq = {}
    for p in paths:
        # Modified to use correct column names and include count column
        df = (pd.read_csv(p, sep="\t", header=None, names=["native", "roman", "count"])
             .dropna(subset=["native", "roman"]).astype(str))
        
        # Debug print to verify data loading
        print(f"Loaded {len(df)} rows from {p}")
        print(f"First 3 rows: {df.head(3)}")
        
        for col in ("native", "roman"):
            for seq in df[col]:
                for ch in seq:
                    freq[ch] = freq.get(ch,0) + 1
    
    # Define explicit special tokens
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(c for c,f in freq.items() if f>=min_freq)
    stoi = {c:i for i,c in enumerate(tokens)}
    itos = {i:c for i,c in enumerate(tokens)}
    return stoi, itos

SRC_STOI, SRC_ITOS = build_vocab([args.train_path, args.dev_path, args.test_path])
TGT_STOI, TGT_ITOS = SRC_STOI, SRC_ITOS

# Define special token constants for clarity
PAD_IDX = SRC_STOI["<PAD>"]
SOS_IDX = SRC_STOI["<SOS>"]
EOS_IDX = SRC_STOI["<EOS>"]
UNK_IDX = SRC_STOI["<UNK>"]

SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = len(SRC_STOI), len(TGT_STOI)
print(f"Vocab sizes → SRC: {SRC_VOCAB_SIZE}, TGT: {TGT_VOCAB_SIZE}")
print(f"Special tokens: PAD={PAD_IDX}, SOS={SOS_IDX}, EOS={EOS_IDX}, UNK={UNK_IDX}")

class TransliterationDataset(Dataset):
    def __init__(self, path, src_stoi, tgt_stoi, max_len=32):
        # Modified to use correct column names and include count column
        df = (pd.read_csv(path, sep="\t", header=None, names=["native", "roman", "count"])
             .dropna(subset=["native", "roman"]).astype(str))
        
        # Make sure we're using the correct columns
        # First column: Bengali script (native/target)
        # Second column: Latin/English transliteration (roman/source)
        self.pairs = df[["native", "roman"]].values.tolist()
        self.src_stoi, self.tgt_stoi, self.max_len = src_stoi, tgt_stoi, max_len
        
        # Debug print to verify data loading
        print(f"Loaded {len(self.pairs)} pairs from {path}")
        if len(self.pairs) > 0:
            print(f"First pair: {self.pairs[0]}")
    
    def __len__(self):
        return len(self.pairs)
    
    def encode_seq(self, seq, stoi):
        seq = str(seq)
        ids = [stoi.get(ch, UNK_IDX) for ch in seq]
        return ids[:self.max_len]
    
    def __getitem__(self, idx):
        # Get native (Bengali) and roman (English) in correct order
        native, roman = self.pairs[idx]
        
        # Roman is source (English input), native is target (Bengali output)
        src_ids = [SOS_IDX] + self.encode_seq(roman, self.src_stoi) + [EOS_IDX]
        tgt_ids = [SOS_IDX] + self.encode_seq(native, self.tgt_stoi) + [EOS_IDX]
        
        # Pad sequences to max length
        pad_src = [PAD_IDX] * ((self.max_len+2) - len(src_ids))
        pad_tgt = [PAD_IDX] * ((self.max_len+2) - len(tgt_ids))
        src_ids += pad_src
        tgt_ids += pad_tgt
        
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# Data augmentation function to reduce overfitting
def augment_data(dataset, augment_factor=0.1):
    """Create augmented versions of the dataset by adding noise"""
    augmented_pairs = []
    
    for native, roman in dataset.pairs:
        # Only augment a fraction of the dataset
        if random.random() > augment_factor:
            continue
            
        # Character deletion (randomly remove a character)
        if len(roman) > 3 and random.random() < 0.3:
            pos = random.randint(0, len(roman)-1)
            aug_roman = roman[:pos] + roman[pos+1:]
            augmented_pairs.append((native, aug_roman))
            
        # Character insertion (duplicate a character)
        if random.random() < 0.3:
            pos = random.randint(0, len(roman)-1)
            aug_roman = roman[:pos] + roman[pos] + roman[pos:]
            augmented_pairs.append((native, aug_roman))
    
    # Add augmented pairs to the dataset
    dataset.pairs.extend(augmented_pairs)
    print(f"Added {len(augmented_pairs)} augmented examples")
    return dataset

train_ds = TransliterationDataset(args.train_path, SRC_STOI, TGT_STOI)
# Apply data augmentation to training set
train_ds = augment_data(train_ds)

dev_ds = TransliterationDataset(args.dev_path, SRC_STOI, TGT_STOI)
test_ds = TransliterationDataset(args.test_path, SRC_STOI, TGT_STOI)

print(f"Dataset sizes → train: {len(train_ds)}, dev: {len(dev_ds)}, test: {len(test_ds)}")

# -----------------------------------
# 3. Model Definitions
# -----------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, cell, drop):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=PAD_IDX)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers, 
                          dropout=drop if n_layers>1 else 0, 
                          batch_first=True)
        # Add embedding dropout to combat overfitting
        self.dropout = nn.Dropout(drop)
        
    def forward(self, src):
        # Apply dropout to embeddings
        emb = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(emb)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, cell, drop):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_IDX)
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.rnn = rnn_cls(emb_dim, hid_dim, n_layers, 
                          dropout=drop if n_layers>1 else 0, 
                          batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        # Add embedding dropout to combat overfitting
        self.dropout = nn.Dropout(drop)
        
    def forward(self, inp, hidden):
        # Apply dropout to embeddings
        emb = self.dropout(self.embedding(inp.unsqueeze(1)))
        out, hidden = self.rnn(emb, hidden)
        return self.fc_out(out.squeeze(1)), hidden

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, decay_rate=0.95, min_tf=0.1, start_tf=1.0):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.decay_rate = decay_rate
        self.min_tf = min_tf
        self.start_tf = start_tf
        
    def forward(self, src, tgt, epoch:int=None, tf_ratio:float=None):
        # scheduled sampling
        if tf_ratio is None:
            tf_ratio = self.start_tf * (self.decay_rate ** (epoch-1)) if epoch else self.start_tf
        tf_ratio = max(tf_ratio, self.min_tf)
        
        bs, tgt_len = tgt.shape
        vocab_size = self.decoder.fc_out.out_features
        
        outputs = torch.zeros(bs, tgt_len, vocab_size, device=src.device)
        
        hidden = self.encoder(src)
        
        # First input to the decoder is the <SOS> token
        inp = tgt[:,0]
        
        for t in range(1, tgt_len):
            out_t, hidden = self.decoder(inp, hidden)
            outputs[:,t] = out_t
            
            # Teacher forcing: use ground truth or predicted token
            if random.random() < tf_ratio:
                inp = tgt[:,t]  # Use ground truth
            else:
                inp = out_t.argmax(1)  # Use model's prediction
                
        return outputs

# Helper function to convert token IDs to string
# ----------------------------------------------------------------
# replace the old ids_to_string with this updated version:
# ----------------------------------------------------------------
def ids_to_string(ids, itos, pad_idx, eos_idx):
    """Convert token IDs to string, stopping at EOS; skip PAD/SOS/UNK."""
    out = []
    for tok in ids:
        # get integer value
        i = tok.item() if isinstance(tok, torch.Tensor) else tok

        # Stop at EOS only
        if i == eos_idx:
            break

        # Skip PAD, SOS, UNK tokens
        if i in [pad_idx, SOS_IDX, UNK_IDX]:
            continue

        # Otherwise append character
        out.append(itos[i])

    return "".join(out)


# -----------------------------------
# 4. Metrics
# -----------------------------------

def calc_word_acc(preds, tgt, pad_idx, eos_idx):
    """Exact-match per sequence word accuracy"""
    bs = tgt.size(0)
    correct = 0
    
    for i in range(bs):
        # gather predicted tokens until eos or pad
        pred_seq = []
        for tok in preds[i].tolist():
            if tok == pad_idx or tok == eos_idx:
                break
            pred_seq.append(tok)
            
        # gold tokens (skip sos)
        tgt_seq = []
        for tok in tgt[i].tolist()[1:]:
            if tok == pad_idx or tok == eos_idx:
                break
            tgt_seq.append(tok)
            
        if pred_seq == tgt_seq:
            correct += 1
            
    return correct / bs

# -----------------------------------
# 5. Train/Eval Loops with Table Visualization
# -----------------------------------

def train_epoch(model, loader, optimizer, criterion, epoch, l2_lambda=1e-5):
    model.train()
    total_loss, corr, count, word_acc_sum = 0, 0, 0, 0
    
    bar = tqdm(loader, desc=f"[Train] Epoch {epoch}", unit="batch")
    
    for src, tgt in bar:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        
        optimizer.zero_grad()
        out = model(src, tgt, epoch=epoch)
        
        loss = criterion(out[:,1:].reshape(-1, out.size(-1)), 
                         tgt[:,1:].reshape(-1))
        
        # Add L2 regularization to combat overfitting
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = out.argmax(2)
        mask = tgt[:,1:] != PAD_IDX
        corr += (preds[:,1:][mask] == tgt[:,1:][mask]).sum().item()
        count += mask.sum().item()
        
        word_acc_sum += calc_word_acc(preds, tgt, pad_idx=PAD_IDX, eos_idx=EOS_IDX)
        
        bar.set_postfix(
            train_loss=total_loss/(bar.n+1),
            char_acc= corr/count,
            word_acc= word_acc_sum/(bar.n+1)
        )
        
    avg_loss = total_loss/len(loader)
    avg_char = corr/count
    avg_word = word_acc_sum/len(loader)
    
    print(f"[Train] Epoch {epoch} → loss: {avg_loss:.4f}, "
          f"char_acc: {avg_char:.4f}, word_acc: {avg_word:.4f}")
    
    return avg_loss, avg_char, avg_word

def eval_epoch(model, loader, criterion, epoch, split="Val", log_table=False):
    model.eval()
    total_loss, corr, count, word_acc_sum = 0, 0, 0, 0
    
    bar = tqdm(loader, desc=f"[{split}] Epoch {epoch}", unit="batch")
    
    # Create W&B table for visualizing predictions
    if log_table:
        table = wandb.Table(columns=["Actual English", "Actual Bengali", "Predicted Bengali", "Correct"])
        samples_to_log = min(100, len(loader.dataset)) # Limit to 100 samples
        sample_indices = random.sample(range(len(loader.dataset)), samples_to_log)
        logged_samples = 0
    
    all_predictions = []  # Store all predictions for saving to file
    
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(bar):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            
            out = model(src, tgt, tf_ratio=0.0)
            
            loss = criterion(out[:,1:].reshape(-1, out.size(-1)), 
                             tgt[:,1:].reshape(-1))
            
            total_loss += loss.item()
            
            preds = out.argmax(2)
            mask = tgt[:,1:] != PAD_IDX
            corr += (preds[:,1:][mask] == tgt[:,1:][mask]).sum().item()
            count += mask.sum().item()
            
            word_acc_sum += calc_word_acc(preds, tgt, pad_idx=PAD_IDX, eos_idx=EOS_IDX)
            
            # Log predictions to W&B table and collect all predictions
            for i in range(src.size(0)):
                global_idx = batch_idx * loader.batch_size + i
                
                # Convert token IDs to strings
                src_str = ids_to_string(src[i], SRC_ITOS, PAD_IDX, EOS_IDX)
                tgt_str = ids_to_string(tgt[i], TGT_ITOS, PAD_IDX, EOS_IDX)
                pred_str = ids_to_string(preds[i], TGT_ITOS, PAD_IDX, EOS_IDX)
                
                # Debug print to check predictions
                if batch_idx == 0 and i < 5:
                    print(f"Example {i}: Input='{src_str}', Target='{tgt_str}', Pred='{pred_str}'")
                
                # Check if prediction is correct
                is_correct = pred_str == tgt_str
                
                # Store all predictions for file output
                all_predictions.append({
                    "source": src_str,
                    "target": tgt_str,
                    "prediction": pred_str if pred_str else "(empty prediction)",
                    "correct": is_correct
                })
                
                # Add to table if this is a sampled example
                if log_table and global_idx in sample_indices:
                    # Add to table
                    table.add_data(src_str, tgt_str, pred_str if pred_str else "(empty prediction)", is_correct)
                    logged_samples += 1
                    
                    # Stop if we've logged enough samples
                    if logged_samples >= samples_to_log:
                        break
            
            bar.set_postfix(**{
                f"{split.lower()}_loss": total_loss/(bar.n+1),
                f"{split.lower()}_char_acc": corr/count,
                f"{split.lower()}_word_acc": word_acc_sum/(bar.n+1)
            })
    
    avg_loss = total_loss/len(loader)
    avg_char = corr/count
    avg_word = word_acc_sum/len(loader)
    
    print(f"[{split}] Epoch {epoch} → loss: {avg_loss:.4f}, "
          f"char_acc: {avg_char:.4f}, word_acc: {avg_word:.4f}")
    
    # Log the table to W&B
    if log_table:
        wandb.log({f"{split.lower()}_predictions": table})
    
    return avg_loss, avg_char, avg_word, all_predictions

# Function to save predictions to file
def save_predictions_to_file(predictions, output_path):
    """Save all predictions to a TSV file"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(predictions)
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {len(predictions)} predictions to {output_path}")

# -----------------------------------
# 6. WandB Sweep
# -----------------------------------

def run_sweep():
    def train_wandb():
        run = wandb.init()
        cfg = run.config
        
        run.name = f"emb{cfg.emb_dim}_hid{cfg.hid_dim}_" \
                  f"{cfg.cell_type}_bs{cfg.batch_size}_lr{cfg.lr}"
        
        use_mem = (DEVICE.type=="cuda")
        
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, 
                                 num_workers=4, pin_memory=use_mem)
        dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, 
                               num_workers=4, pin_memory=use_mem)
        test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, 
                                num_workers=4, pin_memory=use_mem)
        
        enc = Encoder(SRC_VOCAB_SIZE, cfg.emb_dim, cfg.hid_dim, 
                     cfg.n_layers, cfg.cell_type, cfg.dropout)
        dec = Decoder(TGT_VOCAB_SIZE, cfg.emb_dim, cfg.hid_dim, 
                     cfg.n_layers, cfg.cell_type, cfg.dropout)
        
        model = Seq2Seq(enc, dec).to(DEVICE)
        
        if torch.cuda.device_count()>1:
            model = nn.DataParallel(model)
        
        # Use weight decay in optimizer to combat overfitting
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
        
        # Add early stopping to prevent overfitting
        best_val_acc = 0
        patience = cfg.patience
        patience_counter = 0
        best_model_state = None
        
        # Create a table to track predictions over epochs
        epochs_table = wandb.Table(columns=["Epoch", "Example", "English Input", "Bengali Actual", "Bengali Predicted", "Correct"])
        
        # Sample a few examples to track across epochs
        tracking_examples = []
        for _ in range(5): # Track 5 examples
            idx = random.randint(0, len(dev_ds)-1)
            src, tgt = dev_ds[idx]
            tracking_examples.append((idx, src, tgt))
        
        for epoch in range(1, cfg.epochs+1):
            tl, tca, twa = train_epoch(model, train_loader, optimizer, criterion, epoch, l2_lambda=cfg.l2_lambda)
            vl, vca, vwa, _ = eval_epoch(model, dev_loader, criterion, epoch, split="Val", 
                                      log_table=(epoch == cfg.epochs)) # Log table on final epoch
            
            # Log metrics
            wandb.log({
                "epoch": epoch,
                "train_loss": tl, "train_char_acc": tca, "train_word_acc": twa,
                "val_loss": vl, "val_char_acc": vca, "val_word_acc": vwa,
                "gap": tca - vca # Track the gap between train and val accuracy (overfitting measure)
            })
            
            # Early stopping check
            if vca > best_val_acc:
                best_val_acc = vca
                patience_counter = 0
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Track specific examples across epochs
            with torch.no_grad():
                for idx, src, tgt in tracking_examples:
                    src_batch = src.unsqueeze(0).to(DEVICE)
                    tgt_batch = tgt.unsqueeze(0).to(DEVICE)
                    
                    out = model(src_batch, tgt_batch, tf_ratio=0.0)
                    pred = out.argmax(2)[0]
                    
                    # Convert to strings
                    src_str = ids_to_string(src, SRC_ITOS, PAD_IDX, EOS_IDX)
                    tgt_str = ids_to_string(tgt, TGT_ITOS, PAD_IDX, EOS_IDX)
                    pred_str = ids_to_string(pred, TGT_ITOS, PAD_IDX, EOS_IDX)
                    
                    # Check if correct
                    is_correct = pred_str == tgt_str
                    
                    # Add to epochs tracking table
                    epochs_table.add_data(epoch, idx, src_str, tgt_str, 
                                         pred_str if pred_str else "(empty prediction)", is_correct)
        
        # Restore best model for final evaluation
        if best_model_state is not None:
            model.load_state_dict({k: v.to(DEVICE) for k, v in best_model_state.items()})
        
        # Log the epochs tracking table
        wandb.log({"predictions_over_epochs": epochs_table})
        
        # final test with table visualization
        tl, tca, twa, all_predictions = eval_epoch(model, test_loader, criterion, 
                                                 epoch="Final", split="Test", log_table=True)
        
        wandb.log({"test_loss": tl, 
                  "test_char_acc": tca, 
                  "test_word_acc": twa})
        
        print(f">>> Run {run.name} FINAL TEST → "
              f"char_acc: {tca:.4f}, word_acc: {twa:.4f}")
        
        # Save all test predictions to file for GitHub
        output_path = os.path.join(args.predictions_dir, f"test_predictions_{run.name}.tsv")
        save_predictions_to_file(all_predictions, output_path)
        
        wandb.finish()
    
    # sweep_cfg = {
    #     "method":"random",
    #     "metric":{"name":"val_char_acc","goal":"maximize"},
    #     "parameters":{
    #         "emb_dim": {"values":[256, 384]},
    #         "hid_dim": {"values":[256, 384]},
    #         "n_layers": {"values":[1, 2]},
    #         "cell_type": {"values":["RNN"]},  # Added RNN option
    #         "dropout": {"values":[0.2, 0.3, 0.4]},
    #         "lr": {"values":[5e-4, 1e-3]},
    #         "batch_size":{"values":[32, 64]},
    #         "epochs": {"value":10},  # Increased from 1 to 15 for better training
    #         "weight_decay": {"values":[1e-4, 1e-3]},
    #         "l2_lambda": {"values":[1e-5, 1e-4]},
    #         "patience": {"value":5}
    #     }

    sweep_cfg = {
        "method":"random",
        "metric":{"name":"val_char_acc","goal":"maximize"},
        "parameters":{
            "emb_dim": {"values":[256]},
            "hid_dim": {"values":[384]},
            "n_layers": {"values":[2]},
            "cell_type": {"values":["LSTM"]},  # Added RNN option
            "dropout": {"values":[0.3]},
            "lr": {"values":[1e-3]},
            "batch_size":{"values":[64]},
            "epochs": {"value":10},  # Increased from 1 to 15 for better training
            "weight_decay": {"values":[1e-3]},
            "l2_lambda": {"values":[1e-4]},
            "patience": {"value":5}
        }
    }
    
    sweep_id = wandb.sweep(sweep_cfg, project=args.project)
    wandb.agent(sweep_id, train_wandb)

if __name__ == "__main__":
    run_sweep()
