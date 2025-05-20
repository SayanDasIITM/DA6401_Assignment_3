#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Seq2Seq Transliteration with Attention + Rich Metrics + W&B Logging +
Attention Grid + Prediction Table + Connectivity Viz + Test‐set Saving.

Now with:
- Multi‐worker, pinned‐memory data loading
- cuDNN autotuning
- Mixed‐precision (AMP) training
- LR scheduler on plateau
- **Connectivity** visualization (strongest link per output step)
"""
import matplotlib.font_manager as fm  # NEW import

# Load Bengali font
bengali_fp = fm.FontProperties(
    fname=r"C:\Sem-2\DL\Ass-3\kalpurush.ttf"
)

import os, random, tempfile, argparse
import pandas as pd
import editdistance
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from torch.amp import autocast, GradScaler

# ----------------------------------------------------------------------------- 
# 0. cuDNN & GPU setup 
# ----------------------------------------------------------------------------- 
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ----------------------------------------------------------------------------- 
# 1. Arguments 
# ----------------------------------------------------------------------------- 
parser = argparse.ArgumentParser()
parser.add_argument("--train_path", type=str,
    default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.train.tsv")
parser.add_argument("--dev_path", type=str,
    default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.dev.tsv")
parser.add_argument("--test_path", type=str,
    default=r"C:\Sem-2\DL\Ass-3\dakshina_dataset_v1.0\bn\lexicons\bn.translit.sampled.test.tsv")
parser.add_argument("--project", type=str, default="DA6401_Assignment3")
parser.add_argument("--max_len",    type=int,   default=60)
parser.add_argument("--epochs",     type=int,   default=10)
parser.add_argument("--batch_size", type=int,   default=64)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--emb_dim",    type=int,   default=256)
parser.add_argument("--hid_dim",    type=int,   default=512)
parser.add_argument("--dropout",    type=float, default=0.3)
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EXAMPLES = 9  # how many examples for the attention grid each epoch

# ----------------------------------------------------------------------------- 
# 2. Vocab & Dataset 
# ----------------------------------------------------------------------------- 
def build_vocab(paths, min_freq=1):
    freq = {}
    for p in paths:
        df = pd.read_csv(p, sep="\t", header=None, names=["native","roman","count"])\
               .dropna().astype(str)
        for col in ("native","roman"):
            for seq in df[col]:
                for ch in seq:
                    freq[ch] = freq.get(ch,0) + 1

    toks = ["<pad>","<unk>","<sos>","<eos>"] \
        + sorted(c for c,v in freq.items() if v>=min_freq)
    return {c:i for i,c in enumerate(toks)}, {i:c for i,c in enumerate(toks)}

SRC_STOI, SRC_ITOS = build_vocab([args.train_path, args.dev_path, args.test_path])
TGT_STOI, TGT_ITOS = SRC_STOI, SRC_ITOS

class TransliterationDataset(Dataset):
    def __init__(self, path, max_len):
        df = pd.read_csv(path, sep="\t", header=None, 
                         names=["native","roman","count"])\
               .dropna().astype(str)
        self.pairs = df[["native","roman"]].values.tolist()
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def encode(self, seq, stoi):
        return [stoi.get(ch, stoi["<unk>"]) for ch in seq][:self.max_len]

    def __getitem__(self, i):
        nat, rom = self.pairs[i]
        src = [SRC_STOI["<sos>"]] + self.encode(rom, SRC_STOI) + [SRC_STOI["<eos>"]]
        tgt = [TGT_STOI["<sos>"]] + self.encode(nat, TGT_STOI) + [TGT_STOI["<eos>"]]
        pad_s = [SRC_STOI["<pad>"]] * ((self.max_len+2)-len(src))
        pad_t = [TGT_STOI["<pad>"]] * ((self.max_len+2)-len(tgt))
        return (
            torch.tensor(src+pad_s),
            torch.tensor(tgt+pad_t),
            len(src),
            len(tgt)
        )

train_ds = TransliterationDataset(args.train_path, args.max_len)
dev_ds   = TransliterationDataset(args.dev_path,   args.max_len)
test_ds  = TransliterationDataset(args.test_path,  args.max_len)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                          num_workers=2, pin_memory=True)

# ----------------------------------------------------------------------------- 
# 3. Model Definition 
# ----------------------------------------------------------------------------- 
class Encoder(nn.Module):
    def __init__(self,vocab_sz,emb_dim,hid_dim,n_layers,drop):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=SRC_STOI["<pad>"])
        self.rnn = nn.GRU(emb_dim, hid_dim//2, n_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop if n_layers>1 else 0)
    def forward(self,src): 
        return self.rnn(self.emb(src))

class Attention(nn.Module):
    def __init__(self,hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim,1,bias=False)

    def forward(self,hidden,enc_out,mask=None):
        h = hidden[-1].unsqueeze(1).repeat(1,enc_out.size(1),1)
        e = torch.tanh(self.attn(torch.cat([h,enc_out],2)))
        scores = self.v(e).squeeze(2)
        if mask is not None:
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask==0, neg_inf)
        return torch.softmax(scores,1)

class Decoder(nn.Module):
    def __init__(self,vocab_sz,emb_dim,hid_dim,n_layers,drop,attn):
        super().__init__()
        self.emb  = nn.Embedding(vocab_sz, emb_dim, padding_idx=TGT_STOI["<pad>"])
        self.attn = attn
        self.rnn  = nn.GRU(emb_dim+hid_dim, hid_dim, n_layers, batch_first=True)
        self.out  = nn.Linear(hid_dim*2+emb_dim, vocab_sz)
        self.drop = nn.Dropout(drop)

    def forward(self,inp,hidden,enc_out,mask=None):
        emb = self.drop(self.emb(inp).unsqueeze(1))
        a   = self.attn(hidden,enc_out,mask).unsqueeze(1)
        ctx = torch.bmm(a,enc_out)
        out,hidden = self.rnn(torch.cat([emb,ctx],2), hidden)
        out,ctx,emb = out.squeeze(1), ctx.squeeze(1), emb.squeeze(1)
        return self.out(torch.cat([out,ctx,emb],1)), hidden, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder, self.decoder = enc, dec
        self.attn_map = None

    def forward(self, src, src_len, tgt, tf_ratio):
        b, tgt_len = tgt.size()
        vsz = self.decoder.out.out_features
        outputs = torch.zeros(b, tgt_len, vsz, device=src.device)

        enc_out, hidden_enc = self.encoder(src)
        n_layers = hidden_enc.size(0)//2
        hidden_enc = hidden_enc.view(n_layers, 2, b, hidden_enc.size(2))
        hidden = torch.cat([hidden_enc[:,0], hidden_enc[:,1]], dim=2)

        pad_idx = SRC_STOI["<pad>"]
        mask = (src!=pad_idx).to(src.device)

        inp = tgt[:,0]
        attn_maps = []

        for t in range(1, tgt_len):
            pred, hidden, a = self.decoder(inp, hidden, enc_out, mask)
            outputs[:,t] = pred
            attn_maps.append(a)
            inp = tgt[:,t] if random.random()<tf_ratio else pred.argmax(1)

        if attn_maps:
            self.attn_map = torch.stack(attn_maps, dim=1)

        return outputs

def ids_to_string(ids, itos):
    out = []
    for i in ids:
        c = itos[i.item()]
        if c=="<eos>": break
        if c not in ("<pad>","<unk>","<sos>"): out.append(c)
    return "".join(out)

# ----------------------------------------------------------------------------- 
# 4. Training / Eval Loop + AMP + Metrics 
# ----------------------------------------------------------------------------- 
scaler = GradScaler("cuda")

def run_epoch(model, loader, optimizer, crit, epoch, split="train"):
    train_mode = (split=="train")
    model.train() if train_mode else model.eval()

    total_loss = tot_char = corr_char = tot_seq = corr_seq = 0
    char_accs = []

    if args.epochs>1:
        tf = 0.9 - (epoch-1)*(0.4/(args.epochs-1)) if train_mode else 0.0
    else:
        tf = 0.9 if train_mode else 0.0
    tf = max(0.5, tf)

    pbar = tqdm(loader, desc=f"{split} {epoch}")
    for src,tgt,src_len,tgt_len in pbar:
        src,tgt = src.to(DEVICE), tgt.to(DEVICE)
        optimizer.zero_grad()
        with autocast("cuda"):
            out = model(src, src_len, tgt, tf)
            loss = crit(out[:,1:].reshape(-1,out.size(-1)),
                        tgt[:,1:].reshape(-1))
        if train_mode:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        preds = out.argmax(2)

        for i in range(src.size(0)):
            L = tgt_len[i]-1
            tp,tt = preds[i,1:L+1], tgt[i,1:L+1]
            correct = (tp==tt).sum().item()
            corr_char+=correct; tot_char+=L; char_accs.append(correct/L)
            p_str = ids_to_string(preds[i],TGT_ITOS)
            g_str = ids_to_string(tgt[i],TGT_ITOS)
            if p_str==g_str: corr_seq+=1
            tot_seq+=1

    loss    = total_loss/len(loader)
    tok_acc = corr_char/tot_char
    seq_acc = corr_seq/tot_seq
    cpw     = sum(char_accs)/len(char_accs)

    print(f"[{split.capitalize()}] {epoch:2d} "
          f"Loss {loss:.4f} TokAcc {tok_acc:.4f} "
          f"ExactSeq {seq_acc:.4f} CharPerWord {cpw:.4f}")

    wandb.log({
        f"{split}_loss": loss,
        f"{split}_tok_acc": tok_acc,
        f"{split}_exact_seq": seq_acc,
        f"{split}_char_per_word": cpw,
        "epoch": epoch
    }, step=epoch)

    return loss

# ----------------------------------------------------------------------------- 
# 5. Attention Grid + Prediction Table (unchanged) 
# ----------------------------------------------------------------------------- 
def log_attention_and_table(model, dataset, epoch):
    examples = random.sample(dataset.pairs, N_EXAMPLES)
    cols = 3
    rows = (N_EXAMPLES + cols - 1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4))
    axes = axes.flatten()
    table = []

    for idx, (native, roman) in enumerate(examples):
        s_ids = [SRC_STOI["<sos>"]] + [SRC_STOI.get(c, SRC_STOI["<unk>"]) for c in roman] + [SRC_STOI["<eos>"]]
        t_ids = [TGT_STOI["<sos>"]] + [TGT_STOI.get(c, TGT_STOI["<unk>"]) for c in native] + [TGT_STOI["<eos>"]]
        pad_s = [SRC_STOI["<pad>"]] * (args.max_len+2 - len(s_ids))
        pad_t = [TGT_STOI["<pad>"]] * (args.max_len+2 - len(t_ids))
        src_t = torch.tensor([s_ids + pad_s], device=DEVICE)
        tgt_t = torch.tensor([t_ids + pad_t], device=DEVICE)

        with torch.no_grad():
            _ = model(src_t, [len(s_ids)], tgt_t, 0.0)
            attn = model.attn_map[0].cpu().numpy()
            pred = model(src_t, [len(s_ids)], tgt_t, 0.0).argmax(2)[0]
            pred_str = ids_to_string(pred, TGT_ITOS)
            if len(pred_str)==0:
                pred_str="(model needs more training)"

        ax = axes[idx]
        valid_s = min(len(roman)+2, attn.shape[1])
        valid_t = min(len(native)+2, attn.shape[0])
        A = attn[:valid_t, :valid_s]
        A = np.log1p(A * 10)
        A = (A - A.min())/(A.max()-A.min()+1e-9)

        im = ax.imshow(A, aspect='auto', origin='lower',
                       cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(1,valid_s-1))
        ax.set_xticklabels(list(roman), rotation=45, fontsize=12, ha='right')
        ax.set_yticks(range(1, valid_t-1))
        ax.set_yticklabels(list(native), fontproperties=bengali_fp, fontsize=12)
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        ax.set_title(f"Heatmap-{idx+1}", fontsize=14)
        ax.set_xlabel("English Input Characters", fontsize=10)
        ax.set_ylabel("Bengali Output Position", fontsize=10)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label('Attention Weight', rotation=270, labelpad=15)

        table.append([roman, native, pred_str])

    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(pad=2.0)
    fd, fn = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    fig.savefig(fn, dpi=150, bbox_inches='tight')
    plt.close(fig)

    wandb.log({
        "Attention_Heatmaps": wandb.Image(fn),
        "Predictions": wandb.Table(
            data=table,
            columns=["Actual English", "Actual Bengali", "Predicted Bengali"]
        )
    }, step=epoch)
    os.remove(fn)

# ----------------------------------------------------------------------------- 
# 5b. Connectivity Visualization (NEW) 
# ----------------------------------------------------------------------------- 
import torch.nn as nn  # make sure this import sits near the top of your file
def log_connectivity(model, dataset, epoch):
    # ─── 0) Remember original training/eval flag and force train() ───
    was_training = model.training
    model.train()

    # ─── 1) Sample random pair and build src tensor ───
    native, roman = random.choice(dataset.pairs)
    s_ids = [SRC_STOI["<sos>"]] \
          + [SRC_STOI.get(ch, SRC_STOI["<unk>"]) for ch in roman] \
          + [SRC_STOI["<eos>"]]
    pad_s = [SRC_STOI["<pad>"]] * ((args.max_len + 2) - len(s_ids))
    src_indices = torch.tensor([s_ids + pad_s], device=DEVICE)

    # ─── 2) Embed & retain grad ───
    embedded = model.encoder.emb(src_indices)
    embedded.retain_grad()

    # ─── 3) Monkey‐patch encoder.emb ───
    class EmbedWrapper(nn.Module):
        def __init__(self, emb_tensor):
            super().__init__()
            self.embedded = emb_tensor
        def forward(self, x):
            return self.embedded

    orig_emb = model.encoder.emb
    model.encoder.emb = EmbedWrapper(embedded)

    try:
        # ─── 4) Forward pass ───
        out_logits = model(src_indices, [len(s_ids)], src_indices, tf_ratio=0.0)
        valid_t = min(out_logits.size(1), len(native) + 2)

        # ─── Build connectivity table header ───
        table_data = []
        header = ["Bengali →", "English ←"] + ["Strength"]
        # (we'll prepend the actual column names in the WandB.Table call)

        # ─── 5–7) For each output position, backprop and record all grads ───
        for t_out in range(1, valid_t - 1):  # skip <sos> and <eos> positions
            # pick the “correct” target char index
            bchar = native[t_out - 1] if 1 <= t_out - 1 < len(native) else None
            tgt_idx = TGT_STOI.get(bchar, TGT_STOI["<unk>"])

            # zero grads, backprop this single logit
            embedded.grad = None
            logit = out_logits[0, t_out, tgt_idx]
            logit.backward(retain_graph=True)

            # compute per-time-step norms
            grads = embedded.grad[0]                 # (T_in, E)
            conn  = grads.norm(dim=1).cpu().numpy()  # length = src length

            # find the strongest input index
            top_i = int(conn.argmax())

            # label inputs
            inputs = ["⟨sos⟩"] + list(roman) + ["⟨eos⟩"]
            for i, val in enumerate(conn[: len(roman) + 2 ]):
                mark = "✅" if i == top_i else ""
                table_data.append([bchar, f"{inputs[i]} {mark}", f"{val:.4f}"])

    finally:
        # ─── 8) Restore emb layer & mode ───
        model.encoder.emb = orig_emb
        if not was_training:
            model.eval()

    # ─── 9) Log WandB Table ───
    wandb.log({
        "Connectivity": wandb.Table(
            data=table_data,
            columns=["Target Bengali Character",
                     "Input English Character",
                     "Connectivity Strength"]
        )
    }, step=epoch)


# ----------------------------------------------------------------------------- 
# 6. Test‐set Prediction Saver 
# ----------------------------------------------------------------------------- 
def save_test_preds(model, loader):
    """
    Writes out a TSV file with columns:
      English_input    Actual_Bengali    Predicted_Bengali
    """
    os.makedirs("predictions_attention", exist_ok=True)
    out_path = "predictions_attention/test_preds.tsv"
    with open(out_path, "w", encoding="utf-8") as f:
        # Header
        f.write("English_input\tActual_Bengali\tPredicted_Bengali\n")
        model.eval()
        for src, tgt, src_len, tgt_len in tqdm(loader, desc="Writing test preds"):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            with torch.no_grad():
                preds = model(src, src_len, tgt, 0.0).argmax(2)
            for i in range(src.size(0)):
                # Convert index sequences back to strings
                eng_input = ids_to_string(src[i], SRC_ITOS)
                actual_bn = ids_to_string(tgt[i], TGT_ITOS)
                pred_bn   = ids_to_string(preds[i], TGT_ITOS)
                # Write row
                f.write(f"{eng_input}\t{actual_bn}\t{pred_bn}\n")

    print(f">> Wrote {len(loader.dataset)} lines to {out_path}")


# ----------------------------------------------------------------------------- 
# 7. Main Training Loop 
# ----------------------------------------------------------------------------- 
def main():
    enc = Encoder(len(SRC_STOI), args.emb_dim, args.hid_dim, 4, args.dropout)
    att = Attention(args.hid_dim)
    dec = Decoder(len(TGT_STOI), args.emb_dim, args.hid_dim, 4, args.dropout, att)
    model = Seq2Seq(enc, dec).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=TGT_STOI["<pad>"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    wandb.login()
    wandb.init(project=args.project, config=vars(args))

    for ep in range(1, args.epochs+1):
        _ = run_epoch(model, train_loader, optimizer, crit, ep, "train")
        val_loss = run_epoch(model, dev_loader,   optimizer, crit, ep, "val")
        scheduler.step(val_loss)

        log_attention_and_table(model, dev_ds, ep)
        log_connectivity(model,        dev_ds, ep)   # ← NEW!

    run_epoch(model, test_loader, optimizer, crit, args.epochs+1, "test")
    save_test_preds(model, test_loader)
    print("All done.")

if __name__=="__main__":
    main()
