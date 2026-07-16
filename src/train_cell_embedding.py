"""Train on cell-embedding data (experiment #2): each GREmLN pseudocell (pseudobulk
bin) is a sample, patient-grouped folds. Same spirit as the MIL multitask model —
shared encoder (512->hidden, 2 layers) + Thal/Braak heads, combined 1:1 F1, 5 seeds x
grouped folds — but per-bin (no attention pooling, bins are not bagged).

Bins are trained AND scored individually (each bin labelled by its donor's stage) — no
donor aggregation, so F1 is per-pseudocell. Donor grouping is only used for the fold
split (all of a donor's bins stay in one fold) to prevent train/test leakage.

Consumes: data/model_data/<DATA>_<ct>_cellemb_data.pkl (from create_cell_embedding_data.py)

Usage (accepts the MIL command; --mil_hidden_dim maps to the encoder width, and the
MIL/gene-specific flags are parsed-but-ignored since bins are flat 512-vectors):
  python src/train_cell_embedding.py --data SEAAD --cell_type Astrocytes --multitask \
    --mil_hidden_dim 128 --mil_encoder_layers 2 --dropout 0.3441 --weight_decay 0.1 \
    --lr 0.0023849 --batch_size 8 --label_smoothing 0.0 --aug_scale_jitter 0.15 \
    --aug_emb_dim_dropout 0.05 --n_seeds 5 --epochs 200 --patience 40
"""
import argparse
import datetime
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class CellEmbNet(nn.Module):
    def __init__(self, in_dim, hidden, n_layers, dropout, n_thal, n_braak=None):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(max(1, n_layers) - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
        self.enc = nn.Sequential(*layers)
        self.thal_head = nn.Linear(hidden, n_thal)
        self.braak_head = nn.Linear(hidden, n_braak) if n_braak else None

    def forward(self, x):
        h = self.enc(x)
        b = self.braak_head(h) if self.braak_head is not None else None
        return self.thal_head(h), b


def _augment(x, scale_jitter, emb_dim_dropout, noise):
    if scale_jitter > 0:
        u = 1.0 + (torch.rand(x.size(0), 1, device=x.device) * 2 * scale_jitter - scale_jitter)
        x = x * u
    if emb_dim_dropout > 0:
        m = (torch.rand(1, x.size(1), device=x.device) > emb_dim_dropout).float()
        x = x * m
    if noise > 0:
        x = x + torch.randn_like(x) * noise
    return x


def _bin_f1(logits, y_true, idx_to_cls):
    """Per-bin weighted F1: argmax each bin, compare to its (donor's) true label."""
    pred = [int(idx_to_cls[int(i)]) for i in logits.argmax(1).cpu().numpy()]
    truth = [int(v) for v in y_true]
    return f1_score(truth, pred, average="weighted"), truth, pred


def main(args):
    pkl = os.path.join("data", "model_data", f"{args.data}_{args.cell_type}_{args.kind}_data.pkl")
    data = pickle.load(open(pkl, "rb"))
    X_folds = [np.asarray(x, np.float32) for x in data["X_folds"]]
    thal_folds = [np.asarray([int(v) for v in f], np.int64) for f in data["thal_folds"]]
    braak_folds = [np.asarray([int(v) for v in f], np.int64) for f in data["braak_folds"]]
    donor_folds = [np.asarray([str(v) for v in f]) for f in data["donor_folds"]]
    n_folds, D = data["n_folds"], data["embedding_dim"]
    multitask = bool(args.multitask)

    thal_cls = sorted({int(v) for f in thal_folds for v in f})
    thal_to_idx = {c: i for i, c in enumerate(thal_cls)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    braak_cls = sorted({int(v) for f in braak_folds for v in f})
    braak_to_idx = {c: i for i, c in enumerate(braak_cls)}
    idx_to_braak = {i: c for c, i in braak_to_idx.items()}
    n_thal, n_braak = len(thal_cls), (len(braak_cls) if multitask else None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden = int(args.mil_hidden_dim)
    layers = int(args.mil_encoder_layers)
    pca_components = int(args.pca_components) if args.pca_components else 0
    ce = nn.CrossEntropyLoss(label_smoothing=float(args.label_smoothing))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("runs", exist_ok=True)
    logf = open(os.path.join("runs", f"cellemb_{ts}_{args.data}_{args.cell_type}.log"), "w")
    # TensorBoard: same runs/tb root as train_model_v2 so all experiments show together.
    tb_dir = os.path.join("runs", "tb", f"{ts}_{args.data}_{args.cell_type}_{args.kind}_perbin")
    writer = SummaryWriter(tb_dir)

    def out(*a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        logf.write(msg + "\n"); logf.flush()

    out(f"cell-embedding training | {args.data} {args.cell_type} | {sum(len(x) for x in X_folds)} bins, "
        f"{n_folds} grouped folds | Thal {thal_cls} | Braak {braak_cls if multitask else 'off'} | device {device}")
    if not multitask:
        out("  (single-task Thal; pass --multitask for Thal+Braak)")

    all_comb, all_thal_f1, all_braak_f1 = [], [], []
    pooled_thal_t, pooled_thal_p, pooled_braak_t, pooled_braak_p = [], [], [], []

    for seed in range(int(args.n_seeds)):
        torch.manual_seed(seed); np.random.seed(seed)
        seed_comb = []
        for test_fi in range(n_folds):
            tr = [i for i in range(n_folds) if i != test_fi]
            Xtr = np.concatenate([X_folds[i] for i in tr]); ytr_t = np.concatenate([thal_folds[i] for i in tr])
            ytr_b = np.concatenate([braak_folds[i] for i in tr])
            Xte, yte_t, yte_b = X_folds[test_fi], thal_folds[test_fi], braak_folds[test_fi]

            # Per-fold StandardScaler + PCA, fit on TRAIN ONLY (leakage-free; matches
            # flatten_mlp's in-trainer PCA). Data is stored raw so the reduction never
            # sees the held-out fold.
            D_eff = D
            if pca_components and pca_components < Xtr.shape[1]:
                scaler = StandardScaler().fit(Xtr)
                Xtr = scaler.transform(Xtr).astype(np.float32)
                Xte = scaler.transform(Xte).astype(np.float32)
                pca = PCA(n_components=min(pca_components, Xtr.shape[0], Xtr.shape[1]),
                          random_state=seed).fit(Xtr)
                Xtr = pca.transform(Xtr).astype(np.float32)
                Xte = pca.transform(Xte).astype(np.float32)
                D_eff = Xtr.shape[1]

            ytr_ti = torch.tensor([thal_to_idx[int(v)] for v in ytr_t])
            ytr_bi = torch.tensor([braak_to_idx[int(v)] for v in ytr_b])
            loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), ytr_ti, ytr_bi),
                                batch_size=int(args.batch_size) * 16, shuffle=True)
            Xte_t = torch.from_numpy(Xte).to(device)
            # test-fold label indices, for per-epoch test loss (same CE as training)
            yte_ti = torch.tensor([thal_to_idx[int(v)] for v in yte_t]).to(device)
            yte_bi = torch.tensor([braak_to_idx[int(v)] for v in yte_b]).to(device)
            tag = f"s{seed}_f{test_fi + 1}"

            model = CellEmbNet(D_eff, hidden, layers, float(args.dropout), n_thal, n_braak).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

            best_comb, best_thal, best_braak, best_state, no_imp = -1, 0, None, None, 0
            for epoch in range(int(args.epochs)):
                model.train()
                for xb, tb, bb in loader:
                    xb = _augment(xb.to(device), float(args.aug_scale_jitter),
                                  float(args.aug_emb_dim_dropout), float(args.aug_noise_std))
                    tb, bb = tb.to(device), bb.to(device)
                    opt.zero_grad()
                    lt, lb = model(xb)
                    loss = ce(lt, tb) + (float(args.braak_weight) * ce(lb, bb) if multitask else 0.0)
                    loss.backward(); opt.step()
                model.eval()
                with torch.no_grad():
                    lt, lb = model(Xte_t)
                    f1t, _, _ = _bin_f1(lt, yte_t, idx_to_thal)
                    f1b = _bin_f1(lb, yte_b, idx_to_braak)[0] if multitask else None
                    comb = (f1t + f1b) / 2 if multitask else f1t
                    test_loss = ce(lt, yte_ti) + (float(args.braak_weight) * ce(lb, yte_bi)
                                                  if multitask else 0.0)
                writer.add_scalar(f"TestLoss/{tag}", float(test_loss), epoch)
                writer.add_scalar(f"ThalF1/{tag}", f1t, epoch)
                if multitask:
                    writer.add_scalar(f"BraakF1/{tag}", f1b, epoch)
                    writer.add_scalar(f"CombinedF1/{tag}", comb, epoch)
                if comb > best_comb:
                    best_comb, best_thal, best_braak = comb, f1t, f1b
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_imp = 0
                else:
                    no_imp += 1
                    if no_imp >= int(args.patience):
                        break
            model.load_state_dict(best_state)
            # pooled per-bin predictions (at best state)
            with torch.no_grad():
                lt, lb = model(Xte_t)
                _, tt, tp = _bin_f1(lt, yte_t, idx_to_thal)
                pooled_thal_t += tt; pooled_thal_p += tp
                if multitask:
                    _, bt, bp = _bin_f1(lb, yte_b, idx_to_braak)
                    pooled_braak_t += bt; pooled_braak_p += bp
            seed_comb.append(best_comb)
            all_comb.append(best_comb); all_thal_f1.append(best_thal)
            if multitask:
                all_braak_f1.append(best_braak)
            msg = f"  seed {seed} fold {test_fi+1}: Combined={best_comb:.4f} Thal={best_thal:.4f}"
            out(msg + (f" Braak={best_braak:.4f}" if multitask else ""))
        out(f"  Seed {seed} avg combined: {np.mean(seed_comb):.4f}")

    out(f"\n{'#'*56}")
    out(f"Grand Summary: {args.n_seeds} seeds x {n_folds} folds = {len(all_comb)} runs (per-bin F1)")
    if multitask:
        out(f"  Combined (1:1) Generalized F1: {np.mean(all_comb):.4f} +/- {np.std(all_comb):.4f}")
        out(f"    Thal Generalized F1:  {np.mean(all_thal_f1):.4f}")
        out(f"    Braak Generalized F1: {np.mean(all_braak_f1):.4f}")
        pt = f1_score(pooled_thal_t, pooled_thal_p, average="weighted")
        pb = f1_score(pooled_braak_t, pooled_braak_p, average="weighted")
        out(f"  Combined Pooled F1: {(pt+pb)/2:.4f} (Thal {pt:.4f}, Braak {pb:.4f})")
    else:
        out(f"  Thal Generalized F1: {np.mean(all_comb):.4f} +/- {np.std(all_comb):.4f}")
        out(f"  Thal Pooled F1: {f1_score(pooled_thal_t, pooled_thal_p, average='weighted'):.4f}")
    # Summary scalars (step 0) so the generalized numbers show in TB HParams/Scalars.
    writer.add_scalar("CombinedF1/generalized_mean", float(np.mean(all_comb)), 0)
    if multitask:
        writer.add_scalar("ThalF1/generalized_mean", float(np.mean(all_thal_f1)), 0)
        writer.add_scalar("BraakF1/generalized_mean", float(np.mean(all_braak_f1)), 0)
    writer.flush(); writer.close()
    out(f"{'#'*56}")
    out(f"TensorBoard: {tb_dir}")
    logf.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="SEAAD", choices=["SEAAD", "ROSMAP"])
    p.add_argument("--cell_type", default="Astrocytes", choices=["Astrocytes", "Microglia"])
    p.add_argument("--kind", default="cellemb",
                   choices=["cellemb", "patientmean", "pseudobulk", "patientmean_grn", "pseudobulk_grn"],
                   help="Which dataset pickle: cellemb (per-bin emb), patientmean (per-donor "
                        "expr), pseudobulk (per-bin raw expr); *_grn = same on the 3972 GRN genes.")
    p.add_argument("--multitask", action="store_true")
    # encoder / optim (mil_hidden_dim/mil_encoder_layers reused as the encoder width/depth)
    p.add_argument("--mil_hidden_dim", type=int, default=128)
    p.add_argument("--mil_encoder_layers", type=int, default=2)
    p.add_argument("--pca_components", type=int, default=16,
                   help="Per-fold PCA (fit on train only) applied before the encoder; "
                        "0 disables. Data must be stored raw for this to matter.")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--braak_weight", type=float, default=1.0)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=40)
    # augmentation that applies to a flat 512-vec
    p.add_argument("--aug_scale_jitter", type=float, default=0.0)
    p.add_argument("--aug_emb_dim_dropout", type=float, default=0.0)
    p.add_argument("--aug_noise_std", type=float, default=0.0)
    # accepted-but-ignored (don't apply to flat cell embeddings)
    for ign in ["--input_type", "--model_type"]:
        p.add_argument(ign, type=str, default=None)
    for ign in ["--mil_attn_dim", "--mil_cells_per_patient", "--aug_gene_block_cutout_max"]:
        p.add_argument(ign, type=int, default=None)
    p.add_argument("--aug_per_gene_scale_jitter", type=float, default=None)
    main(p.parse_args())
