"""
Run overfitting experiments on **combined** Microglia data: pool CrossRegion, ITG, PFC, then 8-fold CV.

- Combines regions: CrossRegion, ITG, PFC (no EC/V1/V2 for Microglia).
- 8-fold: Stratified split by CERAD. Labels: CERAD + Braak.
- Runs both embedding and expression grids; writes CSVs to --log_base (default: combined_Microglia).
"""
import os
import sys
import pickle
import argparse
import csv
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join("data", "model_data")
MICROGLIA_REGIONS = ["CrossRegion", "ITG", "PFC"]

EXPERIMENTS = [
    {"name": "baseline", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "higher_dropout", "dropout": 0.4, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "highest_dropout", "dropout": 0.5, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "higher_weight_decay", "dropout": 0.2, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "highest_weight_decay", "dropout": 0.2, "weight_decay": 1.0, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "lower_lr", "dropout": 0.2, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "lowest_lr", "dropout": 0.2, "weight_decay": 0.1, "lr": 1e-4, "lambda_entropy": 1e-3},
    {"name": "higher_lambda_entropy", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-2},
    {"name": "highest_lambda_entropy", "dropout": 0.2, "weight_decay": 0.1, "lr": 0.001, "lambda_entropy": 1e-1},
    {"name": "dropout_wd", "dropout": 0.4, "weight_decay": 0.5, "lr": 0.001, "lambda_entropy": 1e-3},
    {"name": "dropout_lr", "dropout": 0.4, "weight_decay": 0.1, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "wd_lr", "dropout": 0.2, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-3},
    {"name": "all_moderate", "dropout": 0.4, "weight_decay": 0.5, "lr": 5e-4, "lambda_entropy": 1e-2},
    {"name": "all_strong", "dropout": 0.5, "weight_decay": 1.0, "lr": 1e-4, "lambda_entropy": 1e-1},
]
GENE_ENCODER_OPTS = [True, False]
ATTENTION_OPTS = [True, False]
PCA_OPTS = [2, 4, 8, 16]
ATTENTION_OPTS_EXPR = [True, False]
PCA_OPTS_EXPR = [2, 4, 8, None]


def load_combined_embedding_data_microglia(regions, n_folds=8):
    """Load Microglia embedding from all regions; patient-level 3D; CERAD from thal, Braak from ptau. Stratified by CERAD."""
    from baseline_model_Microglia import CERAD_CLASS_ORDER, BRAAK_TO_IDX

    cerad_to_idx = {s: i for i, s in enumerate(CERAD_CLASS_ORDER)}
    all_embeddings = []
    all_cerad = []
    all_braak = []

    for region in regions:
        data_path = os.path.join(DATA_DIR, f"{region}_Microglia_data_for_model_training.pkl")
        if not os.path.isfile(data_path):
            print(f"  Skip {region}: not found {data_path}")
            continue
        with open(data_path, "rb") as f:
            embeddings_folds, patient_info_folds, ptau_folds, thal_folds = pickle.load(f)

        for fold_idx in range(len(embeddings_folds)):
            fold_emb = embeddings_folds[fold_idx]
            fold_ptau = ptau_folds[fold_idx]
            fold_thal = thal_folds[fold_idx]
            fold_patient_info = patient_info_folds[fold_idx]
            cell_idx = 0
            for patient_idx, (_, cell_names) in enumerate(fold_patient_info.items()):
                n_c = len(cell_names)
                patient_emb = np.mean(fold_emb[cell_idx : cell_idx + n_c], axis=0)
                all_embeddings.append(patient_emb)
                all_cerad.append(cerad_to_idx.get(str(fold_thal[patient_idx]).strip().lower(), 0))
                all_braak.append(BRAAK_TO_IDX.get(int(round(fold_ptau[patient_idx])), 0))
                cell_idx += n_c

    if not all_embeddings:
        raise FileNotFoundError("No Microglia embedding data loaded from any region.")

    all_embeddings = np.array(all_embeddings)
    all_cerad = np.array(all_cerad)
    all_braak = np.array(all_braak)
    n_cerad_classes = len(CERAD_CLASS_ORDER)
    n_braak_classes = len(BRAAK_TO_IDX)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_patient_data = []
    for _, test_indices in skf.split(np.zeros(len(all_cerad)), all_cerad):
        all_fold_patient_data.append({
            "embeddings": all_embeddings[test_indices],
            "cerad": all_cerad[test_indices],
            "braak": all_braak[test_indices],
        })
    print(f"Combined Microglia embedding: {len(all_embeddings)} patients from {regions}, {n_folds} folds")
    return all_fold_patient_data, n_cerad_classes, n_braak_classes


def load_combined_expression_data_microglia(regions, n_folds=8):
    """Load Microglia raw expression from all regions; CERAD/Braak from thal/ptau; stratified by CERAD."""
    from baseline_model_Microglia import CERAD_CLASS_ORDER, BRAAK_TO_IDX

    cerad_to_idx = {s: i for i, s in enumerate(CERAD_CLASS_ORDER)}
    all_X, all_cerad, all_braak = [], [], []

    for region in regions:
        data_path = os.path.join(DATA_DIR, f"{region}_Microglia_raw_expression_data.pkl")
        if not os.path.isfile(data_path):
            print(f"  Skip {region}: not found {data_path}")
            continue
        with open(data_path, "rb") as f:
            expression_folds, _, ptau_folds, thal_folds = pickle.load(f)
        for fold_idx in range(len(expression_folds)):
            X = np.array(expression_folds[fold_idx], dtype=np.float32)
            ptau, thal = ptau_folds[fold_idx], thal_folds[fold_idx]
            for i in range(len(ptau)):
                all_X.append(X[i])
                all_cerad.append(cerad_to_idx.get(str(thal[i]).strip().lower(), 0))
                all_braak.append(BRAAK_TO_IDX.get(int(round(ptau[i])), 0))

    if not all_X:
        raise FileNotFoundError("No Microglia expression data loaded from any region.")

    all_X = np.array(all_X)
    all_cerad = np.array(all_cerad)
    all_braak = np.array(all_braak)
    n_cerad_classes = len(CERAD_CLASS_ORDER)
    n_braak_classes = len(BRAAK_TO_IDX)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_fold_patient_data = []
    for _, test_indices in skf.split(np.zeros(len(all_cerad)), all_cerad):
        all_fold_patient_data.append({
            "embeddings": all_X[test_indices],
            "cerad": all_cerad[test_indices],
            "braak": all_braak[test_indices],
        })
    print(f"Combined Microglia expression: {len(all_X)} patients from {regions}, {n_folds} folds")
    return all_fold_patient_data, n_cerad_classes, n_braak_classes


def run_embedding_cv_combined_microglia(all_fold_patient_data, n_cerad_classes, n_braak_classes, args):
    """8-fold CV for Microglia embedding. Returns avg_f1 (mean of cerad and braak F1), 0, 0, fold_results."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from train_model_Microglia import EmbeddingDatasetMicroglia, train_epoch_microglia, evaluate_microglia
    from baseline_model_Microglia import BaselineRankingModelMicroglia

    device = torch.device(args.device)
    n_folds = len(all_fold_patient_data)
    criterion_cerad = nn.CrossEntropyLoss()
    criterion_braak = nn.CrossEntropyLoss()

    fold_results = []
    for test_fold_idx in range(n_folds):
        train_idx = [i for i in range(n_folds) if i != test_fold_idx]
        train_emb = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_idx], axis=0)
        train_cerad = np.concatenate([all_fold_patient_data[i]["cerad"] for i in train_idx], axis=0)
        train_braak = np.concatenate([all_fold_patient_data[i]["braak"] for i in train_idx], axis=0)
        test_emb = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_cerad = all_fold_patient_data[test_fold_idx]["cerad"]
        test_braak = all_fold_patient_data[test_fold_idx]["braak"]

        n_train, n_test = train_emb.shape[0], test_emb.shape[0]
        if len(train_emb.shape) == 3:
            if not getattr(args, "gene_encoder", False):
                train_emb = np.mean(train_emb, axis=1)
                test_emb = np.mean(test_emb, axis=1)
        if len(train_emb.shape) == 3:
            n_genes, n_dims = train_emb.shape[1], train_emb.shape[2]
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb.reshape(n_train, -1)).reshape(n_train, n_genes, n_dims)
            test_emb = scaler.transform(test_emb.reshape(n_test, -1)).reshape(n_test, n_genes, n_dims)
            if getattr(args, "pca_components", None) is not None:
                pca = PCA(n_components=args.pca_components)
                train_emb = pca.fit_transform(train_emb.reshape(-1, n_dims)).reshape(n_train, n_genes, args.pca_components)
                test_emb = pca.transform(test_emb.reshape(-1, n_dims)).reshape(n_test, n_genes, args.pca_components)
        else:
            scaler = StandardScaler()
            train_emb = scaler.fit_transform(train_emb)
            test_emb = scaler.transform(test_emb)
            if getattr(args, "pca_components", None) is not None:
                pca = PCA(n_components=args.pca_components)
                train_emb = pca.fit_transform(train_emb)
                test_emb = pca.transform(test_emb)

        train_ds = EmbeddingDatasetMicroglia(train_emb, train_cerad, train_braak)
        test_ds = EmbeddingDatasetMicroglia(test_emb, test_cerad, test_braak)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        if getattr(args, "gene_encoder", False):
            from gene_encoder_model import GeneEncoderModelMicroglia
            n_genes, n_dims = train_emb.shape[1], train_emb.shape[2]
            model = GeneEncoderModelMicroglia(
                n_dims=n_dims, n_cerad_classes=n_cerad_classes, n_braak_classes=n_braak_classes, dropout=args.dropout
            ).to(device)
        else:
            input_shape = (train_emb.shape[1], train_emb.shape[2]) if len(train_emb.shape) == 3 else None
            input_dim = None if input_shape else train_emb.shape[1]
            model = BaselineRankingModelMicroglia(
                input_dim=input_dim, input_shape=input_shape,
                n_cerad_classes=n_cerad_classes, n_braak_classes=n_braak_classes,
                use_attention=getattr(args, "attention", False), num_heads=getattr(args, "attention_heads", 1),
                dropout=args.dropout,
            ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        log_dir = os.path.join(args.log_base, f"combined_Microglia_embedding_{args.exp_id}_testfold{test_fold_idx+1}")
        if os.path.exists(log_dir):
            try:
                import shutil
                shutil.rmtree(log_dir)
            except Exception:
                pass
        writer = SummaryWriter(log_dir)
        best_test_loss = float("inf")
        best_f1 = float("-inf")
        best_state = None
        patience = getattr(args, "patience", 20) or 999
        epochs_without_improvement = 0
        lambda_entropy = getattr(args, "lambda_entropy", 0.0) if getattr(args, "gene_encoder", False) else 0.0

        for epoch in range(args.epochs):
            train_epoch_microglia(model, train_loader, criterion_cerad, criterion_braak, optimizer, device, lambda_entropy=lambda_entropy)
            test_loss, f1_cerad, f1_braak = evaluate_microglia(model, test_loader, criterion_cerad, criterion_braak, device)
            test_f1 = (f1_cerad + f1_braak) / 2.0
            writer.add_scalar("Loss/Test", test_loss, epoch)
            writer.add_scalar("Metrics/Test_F1_cerad", f1_cerad, epoch)
            writer.add_scalar("Metrics/Test_F1_braak", f1_braak, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_f1 = test_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        writer.close()
        if best_state is not None:
            model.load_state_dict(best_state)
        fold_results.append({"best_f1": best_f1})

    return float(np.mean([r["best_f1"] for r in fold_results])), 0.0, 0.0, fold_results


def run_expression_cv_combined_microglia(all_fold_patient_data, n_cerad_classes, n_braak_classes, args):
    """8-fold CV for Microglia expression. Returns avg_f1, 0, 0, fold_results."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    from train_model_Microglia import EmbeddingDatasetMicroglia, train_epoch_microglia, evaluate_microglia
    from baseline_model_Microglia import BaselineRankingModelMicroglia

    device = torch.device(args.device)
    n_folds = len(all_fold_patient_data)
    criterion_cerad = nn.CrossEntropyLoss()
    criterion_braak = nn.CrossEntropyLoss()

    fold_results = []
    for test_fold_idx in range(n_folds):
        train_idx = [i for i in range(n_folds) if i != test_fold_idx]
        train_X = np.concatenate([all_fold_patient_data[i]["embeddings"] for i in train_idx], axis=0)
        train_cerad = np.concatenate([all_fold_patient_data[i]["cerad"] for i in train_idx], axis=0)
        train_braak = np.concatenate([all_fold_patient_data[i]["braak"] for i in train_idx], axis=0)
        test_X = all_fold_patient_data[test_fold_idx]["embeddings"]
        test_cerad = all_fold_patient_data[test_fold_idx]["cerad"]
        test_braak = all_fold_patient_data[test_fold_idx]["braak"]

        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)
        if getattr(args, "pca_components", None) is not None:
            pca = PCA(n_components=args.pca_components)
            train_X = pca.fit_transform(train_X)
            test_X = pca.transform(test_X)

        train_ds = EmbeddingDatasetMicroglia(train_X, train_cerad, train_braak)
        test_ds = EmbeddingDatasetMicroglia(test_X, test_cerad, test_braak)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

        model = BaselineRankingModelMicroglia(
            input_dim=train_X.shape[1], input_shape=None,
            n_cerad_classes=n_cerad_classes, n_braak_classes=n_braak_classes,
            use_attention=False, num_heads=1, dropout=args.dropout,
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        log_dir = os.path.join(args.log_base, f"combined_Microglia_expression_{args.exp_id}_testfold{test_fold_idx+1}")
        if os.path.exists(log_dir):
            try:
                import shutil
                shutil.rmtree(log_dir)
            except Exception:
                pass
        writer = SummaryWriter(log_dir)
        best_test_loss = float("inf")
        best_f1 = float("-inf")
        best_state = None
        patience = getattr(args, "patience", 20) or 999
        epochs_without_improvement = 0

        for epoch in range(args.epochs):
            train_epoch_microglia(model, train_loader, criterion_cerad, criterion_braak, optimizer, device)
            test_loss, f1_cerad, f1_braak = evaluate_microglia(model, test_loader, criterion_cerad, criterion_braak, device)
            test_f1 = (f1_cerad + f1_braak) / 2.0
            writer.add_scalar("Loss/Test", test_loss, epoch)
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_f1 = test_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break
        writer.close()
        if best_state is not None:
            model.load_state_dict(best_state)
        fold_results.append({"best_f1": best_f1})

    return float(np.mean([r["best_f1"] for r in fold_results])), 0.0, 0.0, fold_results


def build_embedding_args(exp, gene_encoder, attention, pca_components, base_args):
    exp_id = f"ge{gene_encoder}_attn{attention}_pca{pca_components}_{exp['name']}"
    return SimpleNamespace(
        region="combined",
        cell_type="Microglia",
        pooling=None if gene_encoder else "mean",
        batch_size=base_args.batch_size,
        epochs=base_args.epochs,
        lr=exp["lr"],
        device=base_args.device,
        dropout=exp["dropout"],
        weight_decay=exp["weight_decay"],
        attention=attention,
        attention_heads=getattr(base_args, "attention_heads", 1),
        pca_components=pca_components,
        gene_encoder=gene_encoder,
        lambda_entropy=exp["lambda_entropy"],
        log_base=base_args.log_base,
        exp_id=exp_id,
        patience=getattr(base_args, "patience", 20),
    )


def build_expression_args(exp, attention, pca_components, base_args):
    exp_id = f"attn{attention}_pca{pca_components}_{exp['name']}"
    return SimpleNamespace(
        region="combined",
        cell_type="Microglia",
        batch_size=base_args.batch_size,
        epochs=base_args.epochs,
        lr=exp["lr"],
        device=base_args.device,
        dropout=exp["dropout"],
        weight_decay=exp["weight_decay"],
        attention=attention,
        attention_heads=getattr(base_args, "attention_heads", 1),
        pca_components=pca_components,
        log_base=base_args.log_base,
        exp_id=exp_id,
        patience=getattr(base_args, "patience", 20),
    )


def main():
    parser = argparse.ArgumentParser(description="Microglia combined (CrossRegion+ITG+PFC) overfitting experiments, 8-fold CV")
    parser.add_argument("--regions", type=str, nargs="*", default=None,
                        help="Regions to combine (default: CrossRegion ITG PFC)")
    parser.add_argument("--n_folds", type=int, default=8, help="Number of CV folds (default: 8)")
    parser.add_argument("--embedding", action="store_true", help="Run embedding experiments")
    parser.add_argument("--expression", action="store_true", help="Run expression experiments")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--attention_heads", type=int, default=1)
    parser.add_argument("--log_base", type=str, default="combined_Microglia", help="Base dir for logs and CSVs")
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    if not args.embedding and not args.expression:
        args.embedding = True
        args.expression = True
    os.makedirs(args.log_base, exist_ok=True)
    regions = args.regions if (args.regions is not None and len(args.regions) > 0) else MICROGLIA_REGIONS
    print(f"Microglia combined regions: {regions}")

    if args.embedding:
        print("\n" + "=" * 70)
        print("Combined Microglia EMBEDDING experiments (8-fold)")
        print("=" * 70)
        try:
            all_fold_data, n_cerad, n_braak = load_combined_embedding_data_microglia(regions, n_folds=args.n_folds)
        except FileNotFoundError as e:
            print(f"Embedding load failed: {e}")
            all_fold_data = None
        if all_fold_data is not None:
            n_runs = len(EXPERIMENTS) * len(GENE_ENCODER_OPTS) * len(ATTENTION_OPTS) * len(PCA_OPTS)
            results = []
            run_idx = 0
            for gene_encoder in GENE_ENCODER_OPTS:
                attn_opts = [False] if gene_encoder else ATTENTION_OPTS
                for attention in attn_opts:
                    for pca_components in PCA_OPTS:
                        for exp in EXPERIMENTS:
                            run_idx += 1
                            run_name = f"ge{gene_encoder}_attn{attention}_pca{pca_components}_{exp['name']}"
                            print(f"\nRun {run_idx}/{n_runs}: {run_name}")
                            train_args = build_embedding_args(exp, gene_encoder, attention, pca_components, args)
                            try:
                                avg_f1, _, _, _ = run_embedding_cv_combined_microglia(all_fold_data, n_cerad, n_braak, train_args)
                            except Exception as e:
                                print(f"  ERROR: {e}")
                                results.append({"region": "combined", "cell_type": "Microglia", "run": run_name,
                                    "experiment": exp["name"], "gene_encoder": gene_encoder, "attention": attention,
                                    "pca_components": pca_components, "dropout": exp["dropout"], "weight_decay": exp["weight_decay"],
                                    "lr": exp["lr"], "lambda_entropy": exp["lambda_entropy"], "avg_f1": None, "error": str(e)})
                                continue
                            results.append({"region": "combined", "cell_type": "Microglia", "run": run_name,
                                "experiment": exp["name"], "gene_encoder": gene_encoder, "attention": attention,
                                "pca_components": pca_components, "dropout": exp["dropout"], "weight_decay": exp["weight_decay"],
                                "lr": exp["lr"], "lambda_entropy": exp["lambda_entropy"], "avg_f1": avg_f1, "error": ""})
            if results:
                out_csv = os.path.join(args.log_base, "overfitting_experiments_combined_embedding.csv")
                fieldnames = ["region", "cell_type", "run", "experiment", "gene_encoder", "attention", "pca_components",
                              "avg_f1", "dropout", "weight_decay", "lr", "lambda_entropy", "error"]
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(results)
                print(f"Wrote {len(results)} rows to {out_csv}")

    if args.expression:
        print("\n" + "=" * 70)
        print("Combined Microglia EXPRESSION experiments (8-fold)")
        print("=" * 70)
        try:
            all_fold_data, n_cerad, n_braak = load_combined_expression_data_microglia(regions, n_folds=args.n_folds)
        except FileNotFoundError as e:
            print(f"Expression load failed: {e}")
            all_fold_data = None
        if all_fold_data is not None:
            n_runs = len(EXPERIMENTS) * len(ATTENTION_OPTS_EXPR) * len(PCA_OPTS_EXPR)
            results = []
            run_idx = 0
            for exp in EXPERIMENTS:
                for attention in ATTENTION_OPTS_EXPR:
                    for pca_components in PCA_OPTS_EXPR:
                        run_idx += 1
                        run_name = f"attn{attention}_pca{pca_components}_{exp['name']}"
                        print(f"\nRun {run_idx}/{n_runs}: {run_name}")
                        train_args = build_expression_args(exp, attention, pca_components, args)
                        try:
                            avg_f1, _, _, _ = run_expression_cv_combined_microglia(all_fold_data, n_cerad, n_braak, train_args)
                        except Exception as e:
                            print(f"  ERROR: {e}")
                            results.append({"region": "combined", "cell_type": "Microglia", "run": run_name,
                                "experiment": exp["name"], "attention": attention, "pca_components": pca_components,
                                "dropout": exp["dropout"], "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                                "lambda_entropy": exp["lambda_entropy"], "avg_f1": None, "error": str(e)})
                            continue
                        results.append({"region": "combined", "cell_type": "Microglia", "run": run_name,
                            "experiment": exp["name"], "attention": attention, "pca_components": pca_components,
                            "dropout": exp["dropout"], "weight_decay": exp["weight_decay"], "lr": exp["lr"],
                            "lambda_entropy": exp["lambda_entropy"], "avg_f1": avg_f1, "error": ""})
            if results:
                out_csv = os.path.join(args.log_base, "overfitting_experiments_combined_expression.csv")
                fieldnames = ["region", "cell_type", "run", "experiment", "attention", "pca_components",
                              "avg_f1", "dropout", "weight_decay", "lr", "lambda_entropy", "error"]
                with open(out_csv, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writeheader()
                    w.writerows(results)
                print(f"Wrote {len(results)} rows to {out_csv}")


if __name__ == "__main__":
    main()
