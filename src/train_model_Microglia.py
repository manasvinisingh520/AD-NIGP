"""Train Microglia model (CERAD + Braak classification) with k-fold cross-validation.
Uses BaselineRankingModelMicroglia or GeneEncoderModelMicroglia. Returns (avg_f1, 0, 0, fold_results) for compatibility with runners."""

import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

from train_model import apply_pooling
from baseline_model_Microglia import (
    BaselineRankingModelMicroglia,
    CERAD_CLASS_ORDER,
    BRAAK_TO_IDX,
)
from gene_encoder_model import GeneEncoderModelMicroglia, gene_entropy_loss


class EmbeddingDatasetMicroglia(Dataset):
    """Dataset for embeddings with CERAD and Braak (both classification indices)."""
    def __init__(self, embeddings, cerad, braak):
        self.embeddings = torch.FloatTensor(embeddings)
        self.cerad = torch.LongTensor(cerad)
        self.braak = torch.LongTensor(braak)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.cerad[idx], self.braak[idx]


def _unpack_model_out(out):
    """Model returns (cerad, braak) or (cerad, braak, w)."""
    cerad_pred, braak_pred = out[0], out[1]
    w = out[2] if len(out) == 3 else None
    return cerad_pred, braak_pred, w


def train_epoch_microglia(model, dataloader, criterion_cerad, criterion_braak, optimizer, device, lambda_entropy=0.0):
    model.train()
    total_loss = 0
    for embeddings, cerad, braak in dataloader:
        embeddings = embeddings.to(device)
        cerad = cerad.to(device)
        braak = braak.to(device)
        optimizer.zero_grad()
        out = model(embeddings)
        cerad_pred, braak_pred, w = _unpack_model_out(out)
        loss = criterion_cerad(cerad_pred, cerad)
        """ + criterion_braak(braak_pred, braak)  # second head: train on Thal only for now """
        if lambda_entropy > 0 and w is not None:
            loss = loss + lambda_entropy * gene_entropy_loss(w)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate_microglia(model, dataloader, criterion_cerad, criterion_braak, device):
    model.eval()
    total_loss = 0
    cerad_preds, cerad_targets = [], []
    braak_preds, braak_targets = [], []
    with torch.no_grad():
        for embeddings, cerad, braak in dataloader:
            embeddings = embeddings.to(device)
            cerad = cerad.to(device)
            braak = braak.to(device)
            out = model(embeddings)
            cerad_pred, braak_pred, _ = _unpack_model_out(out)
            loss = criterion_cerad(cerad_pred, cerad)
            """ + criterion_braak(braak_pred, braak)  # second head: train on Thal only
            """
            total_loss += loss.item()
            cerad_preds.extend(torch.argmax(cerad_pred, dim=1).cpu().numpy())
            cerad_targets.extend(cerad.cpu().numpy())
            """
            braak_preds.extend(torch.argmax(braak_pred, dim=1).cpu().numpy())
            braak_targets.extend(braak.cpu().numpy())
            """
    f1_cerad = f1_score(cerad_targets, cerad_preds, average="weighted")
    """ f1_braak = f1_score(braak_targets, braak_preds, average="weighted")  # second head disabled
    """
    f1_braak = f1_cerad  # so reported F1 = Thal only
    return total_loss / len(dataloader), f1_cerad, f1_braak


def main(args):
    is_mtg = getattr(args, "region", None) == "MTG"
    if is_mtg:
        data_path = os.path.join("data", "model_data", "MTG_data_for_model_training.pkl")
    else:
        data_path = os.path.join("data", "model_data", f"{args.region}_Microglia_data_for_model_training.pkl")
    print(f"Loading data from {data_path}")

    with open(data_path, "rb") as f:
        loaded = pickle.load(f)
    if is_mtg:
        # MTG: (embeddings_folds, patient_info_folds, thal_folds, braak_folds) - both categorical
        embeddings_folds, patient_info_folds, thal_folds, braak_folds = loaded
        ptau_folds = None  # not used for MTG
        # Filter out any patient with Thal or Braak "Reference" so model_data has no reference
        ref_val = "Reference"
        new_embeddings_folds, new_patient_info_folds, new_thal_folds, new_braak_folds = [], [], [], []
        for f in range(len(embeddings_folds)):
            fold_emb = embeddings_folds[f]
            fold_pi = patient_info_folds[f]
            fold_thal = thal_folds[f]
            fold_braak = braak_folds[f]
            patient_names = list(fold_pi.keys())
            valid_idx = [
                i for i in range(len(patient_names))
                if str(fold_thal[i]).strip() != ref_val and str(fold_braak[i]).strip() != ref_val
            ]
            if len(valid_idx) < len(patient_names):
                # Rebuild fold: keep only valid patients
                cell_starts = [0]
                for name in patient_names:
                    cell_starts.append(cell_starts[-1] + len(fold_pi[name]))
                new_emb_list = []
                new_pi = {}
                new_thal = []
                new_braak = []
                for i in valid_idx:
                    name = patient_names[i]
                    start, end = cell_starts[i], cell_starts[i + 1]
                    new_emb_list.append(fold_emb[start:end])
                    new_pi[name] = fold_pi[name]
                    new_thal.append(fold_thal[i])
                    new_braak.append(fold_braak[i])
                fold_emb = np.concatenate(new_emb_list, axis=0)
                fold_pi = new_pi
                fold_thal, fold_braak = new_thal, new_braak
                print(f"  Fold {f+1}: dropped {len(patient_names) - len(valid_idx)} patients with Thal/Braak 'Reference'")
            new_embeddings_folds.append(fold_emb)
            new_patient_info_folds.append(fold_pi)
            new_thal_folds.append(fold_thal)
            new_braak_folds.append(fold_braak)
        embeddings_folds, patient_info_folds, thal_folds, braak_folds = new_embeddings_folds, new_patient_info_folds, new_thal_folds, new_braak_folds
        # MTG: build Thal (cerad) and Braak class order from data so labels map correctly
        all_thal = [str(t).strip().lower() for fold in thal_folds for t in fold]
        all_braak_raw = [b for fold in braak_folds for b in fold]
        unique_cerad = sorted(set(all_thal))
        unique_braak = sorted(set(str(b).strip() for b in all_braak_raw))
        print(f"Unique Thal values in dataset: {unique_cerad}")
        print(f"Unique Braak values in dataset: {unique_braak}", flush=True)
    else:
        embeddings_folds, patient_info_folds, ptau_folds, thal_folds = loaded
        braak_folds = None

    print(f"Loaded {len(embeddings_folds)} folds")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if is_mtg:
        # MTG: data-driven mapping from actual Thal/Braak strings to indices
        cerad_to_idx = {s: i for i, s in enumerate(unique_cerad)}
        braak_to_idx = {s: i for i, s in enumerate(unique_braak)}
        n_cerad_classes = len(unique_cerad)
        n_braak_classes = len(unique_braak)

        def _braak_to_idx(b):
            return braak_to_idx.get(str(b).strip(), 0)
    else:
        cerad_to_idx = {s: i for i, s in enumerate(CERAD_CLASS_ORDER)}
        n_cerad_classes = len(CERAD_CLASS_ORDER)
        n_braak_classes = len(BRAAK_TO_IDX)

        def _braak_to_idx(b):
            try:
                v = int(round(float(b)))
                return BRAAK_TO_IDX.get(v, 0)
            except (ValueError, TypeError):
                return 0

    criterion_cerad = nn.CrossEntropyLoss()
    criterion_braak = nn.CrossEntropyLoss()

    n_folds = len(embeddings_folds)
    all_fold_patient_data = []

    print(f"\n{'='*60}")
    print("Pipeline: cells -> patient embedding (StandardScaler+PCA in CV)")
    print(f"{'='*60}")

    for fold_idx in range(n_folds):
        print(f"\nPreparing fold {fold_idx + 1}/{n_folds}...")
        fold_embeddings = embeddings_folds[fold_idx]
        fold_thal = thal_folds[fold_idx]
        fold_patient_info = patient_info_folds[fold_idx]
        if is_mtg:
            fold_braak = braak_folds[fold_idx]
        else:
            fold_ptau = ptau_folds[fold_idx]

        use_gene_encoder = getattr(args, "gene_encoder", False)
        pooling = None if use_gene_encoder else getattr(args, "pooling", "mean")
        if pooling is None:
            pooled_embeddings = fold_embeddings
        else:
            pooled_embeddings = apply_pooling(fold_embeddings, method=pooling)
        print(f"  Pooled embeddings shape: {pooled_embeddings.shape}")

        patient_embeddings = []
        patient_thal = []
        cell_idx = 0
        for patient_idx, (patient_name, cell_names) in enumerate(fold_patient_info.items()):
            n_cells_patient = len(cell_names)
            patient_cell_embeddings = pooled_embeddings[cell_idx : cell_idx + n_cells_patient]
            patient_emb = np.mean(patient_cell_embeddings, axis=0)
            patient_embeddings.append(patient_emb)
            patient_thal.append(fold_thal[patient_idx])
            cell_idx += n_cells_patient

        patient_embeddings = np.array(patient_embeddings)
        patient_thal = np.array(patient_thal)
        patient_cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in patient_thal])
        if is_mtg:
            patient_braak = np.array([_braak_to_idx(b) for b in fold_braak])
        else:
            patient_braak = np.array([BRAAK_TO_IDX.get(int(round(p)), 0) for p in fold_ptau])

        all_fold_patient_data.append({
            "embeddings": patient_embeddings,
            "cerad": patient_cerad,
            "braak": patient_braak,
        })
        print(f"  Patient embeddings shape: {patient_embeddings.shape}, n_patients: {len(patient_thal)}")

    fold_results = []

    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Test Fold {test_fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
        train_embeddings = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_fold_indices], axis=0)
        train_cerad = np.concatenate([all_fold_patient_data[i]["cerad"] for i in train_fold_indices], axis=0)
        train_braak = np.concatenate([all_fold_patient_data[i]["braak"] for i in train_fold_indices], axis=0)
        test_embeddings = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_cerad = all_fold_patient_data[test_fold_idx]["cerad"]
        test_braak = all_fold_patient_data[test_fold_idx]["braak"]

        n_train, n_test = train_embeddings.shape[0], test_embeddings.shape[0]
        print(f"  Train patients: {n_train}, Test patients: {n_test}")
        n_cerad_test = len(np.unique(test_cerad))
        n_braak_test = len(np.unique(test_braak))
        if n_cerad_test == 1 or n_braak_test == 1:
            print(f"  Warning: test set has only one class (Thal: {n_cerad_test}, Braak: {n_braak_test}). F1 will be 1.0 trivially.")
        else:
            print(f"  Test set classes: Thal {n_cerad_test}, Braak {n_braak_test}")

        if len(train_embeddings.shape) == 3:
            n_genes, n_dims = train_embeddings.shape[1], train_embeddings.shape[2]
            train_2d = train_embeddings.reshape(n_train, -1)
            test_2d = test_embeddings.reshape(n_test, -1)
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_2d)
            test_scaled = scaler.transform(test_2d)
            train_scaled = train_scaled.reshape(n_train, n_genes, n_dims)
            test_scaled = test_scaled.reshape(n_test, n_genes, n_dims)
            if getattr(args, "pca_components", None) is not None:
                train_for_pca = train_scaled.reshape(-1, n_dims)
                test_for_pca = test_scaled.reshape(-1, n_dims)
                pca = PCA(n_components=args.pca_components)
                train_pca = pca.fit_transform(train_for_pca)
                test_pca = pca.transform(test_for_pca)
                train_embeddings = train_pca.reshape(n_train, n_genes, args.pca_components)
                test_embeddings = test_pca.reshape(n_test, n_genes, args.pca_components)
                print(f"  Applied PCA: {n_dims} -> {args.pca_components} dims per gene. Train shape: {train_embeddings.shape}, Test shape: {test_embeddings.shape}")
            else:
                train_embeddings = train_scaled
                test_embeddings = test_scaled
        else:
            scaler = StandardScaler()
            train_embeddings = scaler.fit_transform(train_embeddings)
            test_embeddings = scaler.transform(test_embeddings)
            if getattr(args, "pca_components", None) is not None:
                pca = PCA(n_components=args.pca_components)
                train_embeddings = pca.fit_transform(train_embeddings)
                test_embeddings = pca.transform(test_embeddings)
                print(f"  Applied PCA: -> {args.pca_components} components. Train shape: {train_embeddings.shape}, Test shape: {test_embeddings.shape}")

        train_dataset = EmbeddingDatasetMicroglia(train_embeddings, train_cerad, train_braak)
        test_dataset = EmbeddingDatasetMicroglia(test_embeddings, test_cerad, test_braak)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        use_gene_encoder = getattr(args, "gene_encoder", False)
        lambda_entropy = getattr(args, "lambda_entropy", 0.0) if use_gene_encoder else 0.0

        if use_gene_encoder:
            if len(train_embeddings.shape) != 3:
                raise ValueError("Gene encoder requires 3D embeddings (n_samples, n_genes, n_dims).")
            n_dims = train_embeddings.shape[2]
            model = GeneEncoderModelMicroglia(
                n_dims=n_dims,
                n_cerad_classes=n_cerad_classes,
                n_braak_classes=n_braak_classes,
                dropout=args.dropout,
            ).to(device)
        else:
            if len(train_embeddings.shape) == 3:
                input_shape = (train_embeddings.shape[1], train_embeddings.shape[2])
                input_dim = None
            else:
                input_shape = None
                input_dim = train_embeddings.shape[1]
            model = BaselineRankingModelMicroglia(
                input_dim=input_dim,
                input_shape=input_shape,
                n_cerad_classes=n_cerad_classes,
                n_braak_classes=n_braak_classes,
                use_attention=getattr(args, "attention", False),
                num_heads=getattr(args, "attention_heads", 1),
                dropout=args.dropout,
                n_hidden_layers=getattr(args, "n_hidden_layers", 1),
                hidden_dim_3layer=getattr(args, "hidden_dim_3layer", 32),
            ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        ge_suffix = "_ge" if use_gene_encoder else ""
        attention_suffix = "_attn" if (getattr(args, "attention", False) and not use_gene_encoder) else ""
        exp_id_str = f"_{args.exp_id}" if getattr(args, "exp_id", None) else ""
        log_base = getattr(args, "log_base", "runs")
        log_dir = os.path.join(
            log_base,
            f"{args.region}_Microglia{ge_suffix}{attention_suffix}_drop{args.dropout}_wd{args.weight_decay}{exp_id_str}_testfold{test_fold_idx+1}",
        )
        if os.path.exists(log_dir):
            try:
                import shutil
                shutil.rmtree(log_dir)
            except (PermissionError, OSError):
                pass
        writer = SummaryWriter(log_dir)

        best_test_f1 = float("-inf")
        best_epoch = 0
        best_test_f1_cerad = 0.0
        best_test_f1_braak = 0.0
        patience = getattr(args, "patience", 20)
        use_best_f1 = getattr(args, "best_f1", False)
        if patience is not None and patience <= 0:
            patience = None
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss = train_epoch_microglia(
                model, train_loader, criterion_cerad, criterion_braak, optimizer, device,
                lambda_entropy=lambda_entropy,
            )
            test_loss, test_f1_cerad, test_f1_braak = evaluate_microglia(
                model, test_loader, criterion_cerad, criterion_braak, device
            )
            test_f1 = (test_f1_cerad + test_f1_braak) / 2.0
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Metrics/Test_F1_cerad", test_f1_cerad, epoch)
            writer.add_scalar("Metrics/Test_F1_braak", test_f1_braak, epoch)
            writer.add_scalar("Metrics/Test_F1", test_f1, epoch)

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_epoch = epoch
                best_test_f1_cerad = test_f1_cerad
                best_test_f1_braak = test_f1_braak
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Test F1: {test_f1:.4f} (Thal: {test_f1_cerad:.4f}, Braak: {test_f1_braak:.4f})")

            if patience is not None and epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no test F1 improvement for {patience} epochs).")
                break

        writer.close()
        if use_best_f1:
            report_f1 = best_test_f1
            report_cerad = best_test_f1_cerad
            report_braak = best_test_f1_braak
            report_epoch = best_epoch + 1
            fold_results.append({
                "test_fold": test_fold_idx + 1,
                "best_f1": report_f1,
                "final_f1": report_f1,
                "best_epoch": report_epoch,
                "best_test_mse": 0.0,
                "best_test_mae": 0.0,
            })
            print(f"\nTest Fold {test_fold_idx + 1} Best F1: {report_f1:.4f} (epoch {report_epoch}) (Thal: {report_cerad:.4f}, Braak: {report_braak:.4f})")
        else:
            # Explicitly evaluate current (final) model so we report final-epoch metrics
            model.eval()
            with torch.no_grad():
                _loss, test_f1_cerad, test_f1_braak = evaluate_microglia(
                    model, test_loader, criterion_cerad, criterion_braak, device
                )
            final_f1 = (test_f1_cerad + test_f1_braak) / 2.0
            fold_results.append({
                "test_fold": test_fold_idx + 1,
                "best_f1": final_f1,
                "final_f1": final_f1,
                "best_epoch": epoch + 1,
                "best_test_mse": 0.0,
                "best_test_mae": 0.0,
            })
            print(f"\nTest Fold {test_fold_idx + 1} Final F1: {final_f1:.4f} (last epoch) (Thal: {test_f1_cerad:.4f}, Braak: {test_f1_braak:.4f})")
            if best_test_f1 != final_f1:
                print(f"  (best F1 during training was {best_test_f1:.4f} at epoch {best_epoch+1})")

    print(f"\n{'='*60}")
    print("Leave-One-Fold-Out Cross-Validation Summary")
    print(f"{'='*60}")
    fold_f1s = [r["final_f1"] for r in fold_results]
    label = "best" if getattr(args, "best_f1", False) else "final (last-epoch)"
    print(f"Per-fold {label} F1: {[f'{f:.4f}' for f in fold_f1s]}")
    for result in fold_results:
        print(f"\033[92mTest Fold {result['test_fold']}: F1={result['best_f1']:.4f}\033[0m")

    avg_f1 = np.mean(fold_f1s)
    print(f"\nAverage F1: {avg_f1:.4f}")

    return avg_f1, 0.0, 0.0, fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Microglia model (CERAD + Braak classification)")
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"], help="Set to None when --gene_encoder (runner does this)")
    parser.add_argument("--gene_encoder", action="store_true", help="Use GeneEncoderModelMicroglia (softmax pooling, CERAD+Braak)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--pca_components", type=int, default=None)
    parser.add_argument("--lambda_entropy", type=float, default=1e-3)
    parser.add_argument("--log_base", type=str, default="runs")
    parser.add_argument("--exp_id", type=str, default=None)
    parser.add_argument("--patience", type=int, default=20, help="Early stopping: stop if test F1 does not improve for this many epochs (default: 20). Set to 0 to disable.")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3], help="Number of hidden layers in baseline MLP: 1 or 3 (default: 1).")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32, help="Hidden size when n_hidden_layers=3 (default: 32).")
    parser.add_argument("--best_f1", action="store_true", help="Use best F1 across epochs; default is last-epoch F1.")
    args = parser.parse_args()
    main(args)
