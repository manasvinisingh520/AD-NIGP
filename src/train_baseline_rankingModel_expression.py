"""Train baseline ranking model on patient-level raw expression folds (Astrocytes only, or MTG with CERAD+Braak).

This script reuses the model + training utilities from `src/train_model.py` and `src/baseline_model.py`,
but loads raw expression data produced by `src/create_raw_expression_data.py` with --cell_type Astrocytes or MTG.

Expected pickle format (per region):
    Astrocytes/Microglia: (expression_folds, patient_info_folds, ptau_folds, thal_folds)
    MTG: (expression_folds, patient_info_folds, thal_folds, braak_folds)
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from baseline_model import BaselineRankingModel
from baseline_model_Microglia import (
    BaselineRankingModelMicroglia,
    CERAD_CLASS_ORDER,
    BRAAK_TO_IDX,
)
from train_model import EmbeddingDataset, train_epoch, evaluate
from train_model_Microglia import (
    EmbeddingDatasetMicroglia,
    train_epoch_microglia,
    evaluate_microglia,
)


def _braak_to_idx(b):
    try:
        v = int(round(float(b)))
        return BRAAK_TO_IDX.get(v, 0)
    except (ValueError, TypeError):
        return 0


def _main_mtg(args):
    """MTG expression: load MTG_raw_expression_data.pkl, CERAD + Braak two-head classification."""
    data_path = os.path.join("data", "model_data", "MTG_raw_expression_data.pkl")
    print(f"Loading MTG raw-expression data from {data_path}")
    with open(data_path, "rb") as f:
        expression_folds, patient_info_folds, thal_folds, braak_folds = pickle.load(f)
    # Filter out any patient with Thal or Braak "Reference" so model_data has no reference
    ref_val = "Reference"
    new_expression_folds, new_patient_info_folds, new_thal_folds, new_braak_folds = [], [], [], []
    for f in range(len(expression_folds)):
        fold_X = expression_folds[f]
        fold_pi = patient_info_folds[f]
        fold_thal = thal_folds[f]
        fold_braak = braak_folds[f]
        patient_names = list(fold_pi.keys()) if hasattr(fold_pi, "keys") else list(fold_pi)
        valid_idx = [
            i for i in range(len(patient_names))
            if str(fold_thal[i]).strip() != ref_val and str(fold_braak[i]).strip() != ref_val
        ]
        n_dropped = len(patient_names) - len(valid_idx)
        if n_dropped > 0:
            fold_X = fold_X[valid_idx]
            patient_names = [patient_names[i] for i in valid_idx]
            fold_pi = {n: fold_pi[n] for n in patient_names}
            fold_thal = [fold_thal[i] for i in valid_idx]
            fold_braak = [fold_braak[i] for i in valid_idx]
            print(f"  Fold {f+1}: dropped {n_dropped} patients with Thal/Braak 'Reference'")
        new_expression_folds.append(fold_X)
        new_patient_info_folds.append(fold_pi)
        new_thal_folds.append(fold_thal)
        new_braak_folds.append(fold_braak)
    expression_folds = new_expression_folds
    patient_info_folds = new_patient_info_folds
    thal_folds = new_thal_folds
    braak_folds = new_braak_folds
    n_folds = len(expression_folds)
    print(f"Loaded {n_folds} folds")

    # Unique CERAD (from Thal column) and Braak values in this dataset
    all_thal = [str(t).strip().lower() for fold in thal_folds for t in fold]
    all_braak_raw = [b for fold in braak_folds for b in fold]
    unique_cerad = sorted(set(all_thal))
    unique_braak = sorted(set(str(b).strip() for b in all_braak_raw))
    print(f"Unique Thal values in dataset: {unique_cerad}", flush=True)
    print(f"Unique Braak values in dataset: {unique_braak}", flush=True)
    n_thal_per_fold = [len(f) for f in thal_folds]
    n_braak_per_fold = [len(f) for f in braak_folds]
    print(f"Thal and Braak are per-patient: {n_thal_per_fold} Thal values per fold, {n_braak_per_fold} Braak values per fold (one label per patient)", flush=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # MTG: data-driven Thal/Braak mapping from actual values in the dataset
    cerad_to_idx = {s: i for i, s in enumerate(unique_cerad)}
    braak_to_idx = {s: i for i, s in enumerate(unique_braak)}
    n_cerad_classes = len(unique_cerad)
    n_braak_classes = len(unique_braak)

    def _braak_to_idx_mtg(b):
        return braak_to_idx.get(str(b).strip(), 0)

    criterion_cerad = nn.CrossEntropyLoss()
    criterion_braak = nn.CrossEntropyLoss()

    all_fold_patient_data = []
    for fold_idx in range(n_folds):
        fold_X = np.array(expression_folds[fold_idx], dtype=np.float32)
        fold_thal = thal_folds[fold_idx]
        fold_braak = braak_folds[fold_idx]
        patient_cerad = np.array([cerad_to_idx.get(str(t).strip().lower(), 0) for t in fold_thal], dtype=np.int64)
        patient_braak = np.array([_braak_to_idx_mtg(b) for b in fold_braak], dtype=np.int64)
        all_fold_patient_data.append({"embeddings": fold_X, "cerad": patient_cerad, "braak": patient_braak})
        print(f"Fold {fold_idx + 1}: X={fold_X.shape}, patients={len(fold_thal)}")

    fold_results = []
    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}\nTest Fold {test_fold_idx + 1}/{n_folds}\n{'='*60}")
        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
        train_embeddings = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_fold_indices], axis=0)
        train_cerad = np.concatenate([all_fold_patient_data[i]["cerad"] for i in train_fold_indices], axis=0)
        train_braak = np.concatenate([all_fold_patient_data[i]["braak"] for i in train_fold_indices], axis=0)
        test_embeddings = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_cerad = all_fold_patient_data[test_fold_idx]["cerad"]
        test_braak = all_fold_patient_data[test_fold_idx]["braak"]

        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        test_embeddings = scaler.transform(test_embeddings)
        pca_components = getattr(args, "pca_components", None)
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            train_embeddings = pca.fit_transform(train_embeddings)
            test_embeddings = pca.transform(test_embeddings)

        train_dataset = EmbeddingDatasetMicroglia(train_embeddings, train_cerad, train_braak)
        test_dataset = EmbeddingDatasetMicroglia(test_embeddings, test_cerad, test_braak)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = BaselineRankingModelMicroglia(
            input_dim=train_embeddings.shape[1],
            input_shape=None,
            n_cerad_classes=n_cerad_classes,
            n_braak_classes=n_braak_classes,
            use_attention=False,
            num_heads=1,
            dropout=args.dropout,
            n_hidden_layers=getattr(args, "n_hidden_layers", 1),
            hidden_dim_3layer=getattr(args, "hidden_dim_3layer", 32),
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_base = getattr(args, "log_base", "runs")
        exp_id_str = f"_{args.exp_id}" if getattr(args, "exp_id", None) else ""
        log_dir = os.path.join(log_base, f"MTG_expr{exp_id_str}_testfold{test_fold_idx+1}")
        writer = SummaryWriter(log_dir)

        best_test_f1 = float("-inf")
        best_epoch = 0
        patience = getattr(args, "patience", 20)
        use_best_f1 = getattr(args, "best_f1", False)
        if patience is not None and patience <= 0:
            patience = None
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss = train_epoch_microglia(
                model, train_loader, criterion_cerad, criterion_braak, optimizer, device, lambda_entropy=0.0
            )
            test_loss, test_f1_cerad, test_f1_braak = evaluate_microglia(
                model, test_loader, criterion_cerad, criterion_braak, device
            )
            test_f1 = (test_f1_cerad + test_f1_braak) / 2.0
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Metrics/Test_F1", test_f1, epoch)
            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (epoch + 1) % 10 == 0 and getattr(args, "verbose", True):
                print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Test F1: {test_f1:.4f}")
            if patience is not None and epochs_without_improvement >= patience:
                if getattr(args, "verbose", True):
                    print(f"  Early stopping at epoch {epoch+1} (no test F1 improvement for {patience} epochs).")
                break
        writer.close()
        if use_best_f1:
            print(f"\nTest Fold {test_fold_idx + 1} Best F1: {best_test_f1:.4f} at epoch {best_epoch + 1}")
            fold_results.append({"test_fold": test_fold_idx + 1, "best_f1": best_test_f1, "final_f1": best_test_f1, "best_epoch": best_epoch + 1, "best_test_mse": 0.0, "best_test_mae": 0.0})
        else:
            # Use final-epoch F1 (explicit eval; we never load best state)
            model.eval()
            with torch.no_grad():
                _loss, test_f1_cerad, test_f1_braak = evaluate_microglia(
                    model, test_loader, criterion_cerad, criterion_braak, device
                )
            final_f1 = (test_f1_cerad + test_f1_braak) / 2.0
            print(f"\nTest Fold {test_fold_idx + 1} Final F1: {final_f1:.4f} (last epoch)")
            if best_test_f1 != final_f1:
                print(f"  (best F1 during training was {best_test_f1:.4f} at epoch {best_epoch + 1})")
            fold_results.append({"test_fold": test_fold_idx + 1, "best_f1": final_f1, "final_f1": final_f1, "best_epoch": epoch + 1, "best_test_mse": 0.0, "best_test_mae": 0.0})

    print(f"\n{'='*60}\nLeave-One-Fold-Out Summary (MTG Expression)\n{'='*60}")
    fold_f1s = [r["final_f1"] for r in fold_results]
    label = "best" if getattr(args, "best_f1", False) else "final (last-epoch)"
    print(f"Per-fold {label} F1: {[f'{f:.4f}' for f in fold_f1s]}")
    for r in fold_results:
        print(f"\033[92mTest Fold {r['test_fold']}: F1={r['best_f1']:.4f}\033[0m")
    avg_f1 = np.mean(fold_f1s)
    print(f"\nAverage F1: {avg_f1:.4f}")
    return avg_f1, 0.0, 0.0, fold_results


def main(args):
    if args.region == "MTG":
        return _main_mtg(args)

    data_path = os.path.join(
        "data",
        "model_data",
        f"{args.region}_raw_expression_data.pkl",
    )
    print(f"Loading raw-expression data from {data_path}")

    with open(data_path, "rb") as f:
        expression_folds, patient_info_folds, ptau_folds, thal_folds = pickle.load(f)

    n_folds = len(expression_folds)
    print(f"Loaded {n_folds} folds")

    if args.attention:
        print("Note: --attention is ignored for raw expression data (1D input); attention only applies to 2D (n_genes, n_dims) inputs.")

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Loss functions
    if args.loss_ptau is None or (isinstance(args.loss_ptau, str) and args.loss_ptau.lower() == "none"):
        criterion_ptau = None
    elif args.loss_ptau == "mse":
        criterion_ptau = nn.MSELoss()
    elif args.loss_ptau == "mae":
        criterion_ptau = nn.L1Loss()
    elif args.loss_ptau == "huber":
        criterion_ptau = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss_ptau: {args.loss_ptau}")

    criterion_thal = nn.CrossEntropyLoss()

    # Thal mapping across all folds (manual mapping, consistent with baseline script)
    all_thal_labels = []
    for thal_fold in thal_folds:
        all_thal_labels.extend(thal_fold)
    all_thal_labels_int = [int(x) for x in all_thal_labels]
    thal_classes = sorted(set(all_thal_labels_int))
    thal_to_idx = {c: i for i, c in enumerate(thal_classes)}
    idx_to_thal = {i: c for c, i in thal_to_idx.items()}
    n_thal_classes = len(thal_classes)

    print(f"Thal classes (sorted): {thal_classes}")
    print(f"Thal to index mapping: {thal_to_idx}")

    # Prepare fold data (already patient-level)
    all_fold_patient_data = []
    for fold_idx in range(n_folds):
        fold_X = np.array(expression_folds[fold_idx], dtype=np.float32)  # (n_patients, n_genes)
        fold_ptau = np.array(ptau_folds[fold_idx], dtype=np.float32)
        fold_thal_int = [int(t) for t in thal_folds[fold_idx]]
        fold_thal = np.array([thal_to_idx[t] for t in fold_thal_int], dtype=np.int64)

        # log1p ptau target
        fold_ptau = np.log1p(fold_ptau)

        # keep patient_info_folds for debugging/order checks
        _fold_patient_info = patient_info_folds[fold_idx]
        if len(_fold_patient_info) != len(fold_X):
            print(
                f"Warning: fold {fold_idx}: patient_info has {len(_fold_patient_info)} patients, "
                f"but X has {len(fold_X)} rows. Ensure create_raw_expression_data.py preserves order."
            )

        all_fold_patient_data.append({"embeddings": fold_X, "ptau": fold_ptau, "thal": fold_thal})

        print(f"Fold {fold_idx + 1}: X={fold_X.shape}, patients={len(fold_ptau)}")

    # Leave-one-fold-out CV
    fold_results = []
    for test_fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Test Fold {test_fold_idx + 1}/{n_folds}")
        print(f"Training on folds: {[i+1 for i in range(n_folds) if i != test_fold_idx]}")
        print(f"{'='*60}")

        train_fold_indices = [i for i in range(n_folds) if i != test_fold_idx]
        train_embeddings = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_fold_indices], axis=0)
        train_ptau = np.concatenate([all_fold_patient_data[i]["ptau"] for i in train_fold_indices], axis=0)
        train_thal = np.concatenate([all_fold_patient_data[i]["thal"] for i in train_fold_indices], axis=0)

        test_embeddings = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_ptau = all_fold_patient_data[test_fold_idx]["ptau"]
        test_thal = all_fold_patient_data[test_fold_idx]["thal"]

        # StandardScaler normalization (fit on train, apply to train + test)
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        test_embeddings = scaler.transform(test_embeddings)
        print("Applied StandardScaler normalization (fit on train, applied to train and test)")

        pca_components = getattr(args, "pca_components", None)
        if pca_components is not None:
            pca = PCA(n_components=pca_components)
            train_embeddings = pca.fit_transform(train_embeddings)
            test_embeddings = pca.transform(test_embeddings)
            print(f"Applied PCA: reduced to {pca_components} components (fit on train, transform train and test)")

        print(f"Train: {len(train_embeddings)} patients, Test: {len(test_embeddings)} patients")

        train_dataset = EmbeddingDataset(train_embeddings, train_ptau, train_thal)
        test_dataset = EmbeddingDataset(test_embeddings, test_ptau, test_thal)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = BaselineRankingModel(
            input_dim=train_embeddings.shape[1],
            input_shape=None,
            n_thal_classes=n_thal_classes,
            use_attention=args.attention,
            num_heads=args.attention_heads if args.attention else 1,
            dropout=args.dropout,
            n_hidden_layers=args.n_hidden_layers,
            hidden_dim_3layer=args.hidden_dim_3layer,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_base = getattr(args, "log_base", "runs")
        attention_suffix = "_attn" if args.attention else ""
        dropout_str = f"_drop{args.dropout}"
        wd_str = f"_wd{args.weight_decay}"
        layers_str = f"_L{args.n_hidden_layers}" if args.n_hidden_layers == 3 else ""
        pca_str = f"_pca{pca_components}" if pca_components is not None else "_pcaNone"
        exp_id_str = f"_{args.exp_id}" if getattr(args, "exp_id", None) else ""
        # When exp_id is set (e.g. attnTrue_huber_pca2_baseline), use it instead of pca_str to match embedding-style path
        suffix = f"{exp_id_str}" if exp_id_str else f"{layers_str}{pca_str}"
        log_dir = os.path.join(
            log_base,
            f"{args.region}_Astrocytes_expr_{args.loss_ptau}{attention_suffix}{dropout_str}{wd_str}{suffix}_testfold{test_fold_idx+1}",
        )
        writer = SummaryWriter(log_dir)

        best_test_loss = float("inf")
        best_test_f1 = float("-inf")
        best_epoch = 0
        best_test_mse = None
        best_test_mae = None
        best_state = None
        patience = getattr(args, "patience", 20)
        use_best_f1 = getattr(args, "best_f1", False)
        if patience is not None and patience <= 0:
            patience = None
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_loss, train_ptau_loss, train_thal_loss = train_epoch(
                model, train_loader, criterion_ptau, criterion_thal, optimizer, device
            )
            test_loss, test_mse, test_mae, test_f1 = evaluate(
                model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal
            )

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Loss/Train_Ptau", train_ptau_loss, epoch)
            writer.add_scalar("Loss/Train_Thal", train_thal_loss, epoch)
            writer.add_scalar("Metrics/Test_MSE", test_mse, epoch)
            writer.add_scalar("Metrics/Test_MAE", test_mae, epoch)
            writer.add_scalar("Metrics/Test_F1", test_f1, epoch)

            if use_best_f1:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_test_f1 = test_f1
                    best_epoch = epoch
                    best_test_mse = test_mse
                    best_test_mae = test_mae
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            else:
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            """
            # (legacy: best_f1 controlled by --best_f1 above)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_f1 = test_f1
                best_epoch = epoch
                best_test_mse = test_mse
                best_test_mae = test_mae
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            """

            if (epoch + 1) % 10 == 0 and getattr(args, "verbose", True):
                print(
                    f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}, Test MAE: {test_mae:.4f}"
                )

            if patience is not None and epochs_without_improvement >= patience:
                if getattr(args, "verbose", True):
                    print(f"  Early stopping at epoch {epoch+1} (no test loss improvement for {patience} epochs).")
                break

        """
        if best_state is not None:
            model.load_state_dict(best_state)
        writer.close()
        print(f"\nTest Fold {test_fold_idx + 1} Best F1: {best_test_f1:.4f} at epoch {best_epoch + 1}")
        print(f"  MSE at best F1: {best_test_mse:.4f}, MAE at best F1: {best_test_mae:.4f}")
        fold_results.append(
            {
                "test_fold": test_fold_idx + 1,
                "best_f1": best_test_f1,
                "best_epoch": best_epoch + 1,
                "best_test_mse": best_test_mse,
                "best_test_mae": best_test_mae,
            }
        )
        """
        if use_best_f1 and best_state is not None:
            model.load_state_dict(best_state)
        writer.close()

        if use_best_f1 and best_state is not None:
            report_f1 = best_test_f1
            report_mse = best_test_mse
            report_mae = best_test_mae
            report_epoch = best_epoch + 1
            print(f"\nTest Fold {test_fold_idx + 1} Best F1: {report_f1:.4f} (epoch {report_epoch})")
            print(f"  MSE: {report_mse:.4f}, MAE: {report_mae:.4f}")
            fold_results.append(
                {
                    "test_fold": test_fold_idx + 1,
                    "best_f1": report_f1,
                    "final_f1": report_f1,
                    "best_epoch": report_epoch,
                    "best_test_mse": report_mse,
                    "best_test_mae": report_mae,
                }
            )
        else:
            # Final-epoch F1/MSE/MAE (explicit eval)
            model.eval()
            with torch.no_grad():
                test_loss, test_mse, test_mae, test_f1 = evaluate(
                    model, test_loader, criterion_ptau, criterion_thal, device, idx_to_thal
                )
            report_f1 = test_f1
            report_mse = test_mse
            report_mae = test_mae
            report_epoch = epoch + 1
            print(f"\nTest Fold {test_fold_idx + 1} Final F1: {report_f1:.4f} (last epoch)")
            if use_best_f1:
                print(f"  (no improvement in any epoch; reporting last epoch)")
            print(f"  MSE: {report_mse:.4f}, MAE: {report_mae:.4f}")
            fold_results.append(
                {
                    "test_fold": test_fold_idx + 1,
                    "best_f1": report_f1,
                    "final_f1": report_f1,
                    "best_epoch": report_epoch,
                    "best_test_mse": report_mse,
                    "best_test_mae": report_mae,
                }
            )

    print(f"\n{'='*60}")
    print("Leave-One-Fold-Out Cross-Validation Summary (Expression)")
    print(f"{'='*60}")
    fold_f1s = [r["final_f1"] for r in fold_results]
    label = "best" if getattr(args, "best_f1", False) else "final (last-epoch)"
    print(f"Per-fold {label} F1: {[f'{f:.4f}' for f in fold_f1s]}")
    for result in fold_results:
        print(
            f"\033[92mTest Fold {result['test_fold']}: F1={result['best_f1']:.4f}, "
            f"MSE={result['best_test_mse']:.4f}, MAE={result['best_test_mae']:.4f}\033[0m"
        )

    avg_f1 = np.mean(fold_f1s)
    avg_mse = np.mean([r["best_test_mse"] for r in fold_results])
    avg_mae = np.mean([r["best_test_mae"] for r in fold_results])

    print(f"\nAverage F1: {avg_f1:.4f}")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")

    return avg_f1, avg_mse, avg_mae, fold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline ranking model on raw expression folds (Astrocytes only)")
    parser.add_argument("--region", type=str, required=True, help="Region name (e.g., EC, ITG, PFC, MTG). MTG uses CERAD+Braak two-head model.")
    parser.add_argument(
        "--loss_ptau",
        type=str,
        default="mse",
        choices=["mse", "mae", "huber", "none"],
        help="Loss function for ptau prediction (none to omit ptau loss)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay for Adam optimizer (default: 1e-2)")
    parser.add_argument("--attention", action="store_true", help="Use self-attention layer (ignored for expression data: 1D input)")
    parser.add_argument("--attention_heads", type=int, default=1, help="Number of attention heads when --attention is set (default: 1)")
    parser.add_argument("--pca_components", type=int, default=None, help="PCA components (None = no PCA). Reduces n_genes to this many dimensions.")
    parser.add_argument("--log_base", type=str, default="runs", help="Base directory for TensorBoard logs (default: runs)")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print per-epoch progress (default: True).")
    parser.add_argument("--no_verbose", action="store_false", dest="verbose", help="Disable per-epoch printing (quiet mode for batch runs)")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping: stop if test loss does not improve for this many epochs (default: 20). Set to 0 to disable.")
    parser.add_argument("--n_hidden_layers", type=int, default=1, choices=[1, 3], help="Number of hidden layers in baseline MLP: 1 (minimal) or 3 (original). Default: 1.")
    parser.add_argument("--hidden_dim_3layer", type=int, default=32, help="Hidden size when n_hidden_layers=3 (default: 32).")

    parser.add_argument("--best_f1", action="store_true", help="Use best F1 across epochs and restore best model; default is last-epoch F1.")

    main(parser.parse_args())

