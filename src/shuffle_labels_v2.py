"""Create a label-shuffled (permutation-control) overlay for a v2 data pickle.

Negative control for the DEFAULT pipeline (train_model_v2 on the v2 data). Permutes
the per-donor Thal and Braak labels across individuals so the marginal stage
distributions are exactly preserved, but each donor's expression is now paired with a
different donor's labels (breaking the expression->label link).

  JOINT (default):  each donor is reassigned another donor's (Thal, Braak) PAIR -> keeps
                    both marginals and the Thal-Braak correlation.
  --independent:    Thal and Braak permuted separately -> also decorrelates them.

Saves only the shuffled fold-labels (a small overlay), NOT a copy of the 5.8 GB tensors.
Apply it at train time with:  train_model_v2.py ... --shuffle_labels <overlay.pkl>

Usage:
  python src/shuffle_labels_v2.py --data SEAAD --cell_type Microglia --seed 0
"""
import argparse
import os
import pickle
from collections import Counter

import numpy as np

from train_model_v2 import _load_v2_pickle


def _reshape(flat, sizes):
    out, k = [], 0
    for s in sizes:
        out.append(list(flat[k:k + s]))
        k += s
    return out


def main(args):
    data, ct = args.data, args.cell_type
    d = _load_v2_pickle(ct if data == "Atlas" else args.region or ct, ct,
                        expr_transform=args.expr_transform, dataset=data)
    thal_folds = [[int(float(x)) for x in f] for f in d["thal_folds"]]
    braak_raw = d.get("braak_folds")
    braak_folds = [[int(float(x)) for x in f] for f in braak_raw] if braak_raw else None
    if not braak_folds or not any(len(f) for f in braak_folds):
        raise SystemExit(f"{data} {ct} has no Braak labels -- cannot shuffle both.")

    t_sizes = [len(f) for f in thal_folds]
    b_sizes = [len(f) for f in braak_folds]
    if t_sizes != b_sizes:
        raise SystemExit(f"Thal/Braak per-fold sizes differ ({t_sizes} vs {b_sizes}); "
                         "joint donor shuffle needs them aligned.")
    thal_flat = [x for f in thal_folds for x in f]
    braak_flat = [x for f in braak_folds for x in f]
    n = len(thal_flat)
    print(f"{data} {ct}: {n} donors across {len(thal_folds)} folds")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    if args.independent:
        perm2 = rng.permutation(n)
        new_thal = [thal_flat[i] for i in perm]
        new_braak = [braak_flat[i] for i in perm2]
        mode = "independent"
    else:
        new_thal = [thal_flat[i] for i in perm]
        new_braak = [braak_flat[i] for i in perm]
        mode = "joint"

    # --- Verify: marginals unchanged, labels actually moved ---
    assert Counter(new_thal) == Counter(thal_flat), "Thal marginal changed!"
    assert Counter(new_braak) == Counter(braak_flat), "Braak marginal changed!"
    thal_changed = sum(a != b for a, b in zip(thal_flat, new_thal))
    braak_changed = sum(a != b for a, b in zip(braak_flat, new_braak))
    print(f"  mode={mode} seed={args.seed}")
    print(f"  Thal  marginal preserved; {thal_changed}/{n} donors' Thal changed")
    print(f"  Braak marginal preserved; {braak_changed}/{n} donors' Braak changed")
    print(f"  Thal dist : {dict(sorted(Counter(thal_flat).items()))}")
    print(f"  Braak dist: {dict(sorted(Counter(braak_flat).items()))}")

    overlay = {
        "thal_folds": _reshape(new_thal, t_sizes),
        "braak_folds": _reshape(new_braak, b_sizes),
        "n_folds": len(thal_folds),
        "data": data, "cell_type": ct, "expr_transform": args.expr_transform,
        "seed": args.seed, "mode": mode, "kind": "label_shuffle_overlay",
    }
    out = args.output or os.path.join(
        "data", "model_data", f"{data}_{ct}_v2_data_{args.expr_transform}_labelshuffle_{mode}_seed{args.seed}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(overlay, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved overlay -> {out}")
    print(f"Train with:  --shuffle_labels {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="SEAAD", choices=["SEAAD", "ROSMAP"])
    p.add_argument("--cell_type", default="Microglia", choices=["Astrocytes", "Microglia"])
    p.add_argument("--region", default=None, help="Label only (SEAAD/ROSMAP load by cell_type).")
    p.add_argument("--expr_transform", default="log1p")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--independent", action="store_true",
                   help="Shuffle Thal and Braak separately (also decorrelates them).")
    p.add_argument("--output", default=None)
    main(p.parse_args())
