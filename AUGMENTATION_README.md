# Training augmentations (plain English)

This document describes every augmentation-related knob used by **`train_model.py`** when learning from **patient-level embeddings**. The goal is to make the model less sensitive to exact input values and to improve generalization—but turning things up too high can also make learning harder.

**Important rules up front:**

- **Training only:** All of the `--aug_*` options below are applied **only on training batches**. Validation and test folds see the **original** embeddings.
- **Two “stages” of augmentation:** Some changes happen **when a single patient’s embedding is loaded** (the `--aug_*` flags). Others happen **inside each training batch**, by mixing **pairs of patients** in that batch (`mixup` and `gene CutMix`).
- **2D vs 3D embeddings:** After pooling, each patient may be a **flat vector** (2D pipeline: many features). With **no pooling** or **gene-encoder** inputs, each patient is a **genes × dimensions** matrix (3D pipeline). A few options apply only in one case; that is called out below.
- **Ordinal Thalamus mode:** If you use **`--thal_ordinal`**, **batch mixup** and **gene CutMix are turned off** (they do not mix cleanly with the ordinal loss). Dataset-level `--aug_*` can still run unless you disable them separately.

---

## Master switch

### `--no_online_augment`

When this flag is set, **every `--aug_*` dataset augmentation is disabled** (all treated as zero/off). It does **not** by itself change `mixup` or `gene_cutmix_*`; those are separate flags and are still governed by their numeric values and by `--thal_ordinal`.

---

## Dataset augmentations (applied per sample when it is loaded for training)

These run **in a fixed order** for each training example: **random scaling first**, then **random masks (dropout / cutout)**, then **Gaussian noise**.

### `--aug_scale_jitter`

**What it does:** Multiplies the **entire** embedding (all values) by **one random number** for that sample. The number is drawn uniformly between **1 − a** and **1 + a**, where **a** is the value you pass (the “half width”).

**Plain English:** Randomly makes the whole vector a bit brighter or dimmer, like turning a volume knob for that patient once per load.

**Typical use:** Small positive values add mild variety without changing which genes or dimensions dominate.

---

### `--aug_per_gene_scale_jitter`

**What it does:** Same idea as scale jitter, but **each gene row** (in the 3D case) gets **its own** random multiplier in **1 − b … 1 + b**, where **b** is your value.

**Plain English:** Some genes get amplified, others damped, independently. It does **not** apply to the fully pooled 1D feature vector path the same way as the gene matrix path (see code: it applies when the sample is still shaped as genes × dimensions).

**Typical use:** Encourages the model not to rely on every gene having an exact fixed scale.

---

### `--aug_gene_dropout`

**What it does:** For **3D** inputs (genes × dimensions), each **entire gene row** is set to zero with probability equal to your value (like dropout over genes).

**Plain English:** Randomly “forgets” whole genes for that patient on that pass, so the network must not depend on every single gene always being present.

**Note:** On **2D pooled** inputs there is no gene axis in the same sense; this option targets the gene-matrix view.

---

### `--aug_emb_dim_dropout`

**What it does:** For **3D** inputs, each **embedding dimension** (the same index across all genes) is **zeroed out** for all genes at once with the probability you set—one mask shared across the gene axis for that dimension. For **2D / flat** features it applies along the **feature** dimension in an analogous way.

**Plain English:** Randomly removes the same “component direction” everywhere, so the model cannot rely on one synthetic dimension always being visible.

---

### `--aug_feature_dropout`

**What it does:** Intended for **flat (1D)** feature vectors after pooling. Each **feature** is zeroed independently with the probability you set.

**Plain English:** Similar to classic dropout on input features—randomly blanks scattered scalars in the pooled vector.

---

### `--aug_gene_block_cutout_max` and `--aug_gene_block_cutout_prob`

**What they do:** For **3D** gene matrices, if the max length is **greater than zero**, then with probability **`aug_gene_block_cutout_prob`** the code picks a **contiguous run of genes** along the list and sets **all their rows** to zero. The **length** of that run is random between **1** and **min(max, number of genes)**.

**Plain English:** Sometimes wipes out a **chunk of neighboring genes** in fixed gene order (not biological proximity, just index order in the tensor). That forces the model to work when a whole segment of the input is missing.

**Note:** If `aug_gene_block_cutout_max` is **0**, this is **off** regardless of probability.

---

### `--aug_noise_std`

**What it does:** Adds **independent Gaussian noise** to **every** entry of the embedding, with standard deviation equal to the value you pass.

**Plain English:** Jitters every number a little so tiny numerical quirks matter less. Too large, and you drown the signal.

---

## Batch-level mixing (two patients in the same minibatch)

These are **not** `--aug_*` flags. They run inside the training step when the batch has **more than one** patient and **ordinal Thalamus loss is off**.

### `--mixup_alpha`

**What it does:** With probability driven by whether alpha is **> 0** (and when CutMix does not run first—see code), the code blends **two random patients in the batch** in embedding space. A mixing weight **λ** is sampled from a **Beta(α, α)** distribution. Embeddings become **λ A + (1−λ) B**, and **p-tau targets** are blended the same way. The **classification loss** is also averaged using **λ** so supervision matches the mixed inputs.

**Plain English:** Creates synthetic “in-between” patients by averaging examples; the labels are averaged too so the training target still matches what the network is asked to fit.

**α = 1** means λ is **uniform** between 0 and 1. **Larger α** pushes λ toward **one half** (more balanced mixes). **Smaller α** favors **extreme** mixes (very close to one parent or the other). **0** means **off**.

---

### `--gene_cutmix_prob` and `--gene_cutmix_alpha`

**What they do:** Only for **3D** gene matrices and when a random draw is below **`gene_cutmix_prob`**, the code picks a **partner patient** in the batch, samples a **fraction** of genes from a **Beta(α, α)** distribution, turns that into a **contiguous block length**, picks a **random starting gene**, and **replaces that block** in patient A with the **same block** from patient B. **p-tau** is mixed with weight **λ = (block length) / (total genes)**. The **Thalamus classification loss** is mixed in a way that matches this cut (the partner supplies labels for the cut region).

**Plain English:** Imagine splicing a stretch of consecutive genes from another patient into the current matrix—like a localized transplant—while mixing regression targets and classification loss accordingly.

**`gene_cutmix_prob`** is simply **how often** this happens per batch when conditions allow. **`gene_cutmix_alpha`** only shapes **how wide** the swapped stretch tends to be (same Beta intuition as mixup: α = 1 → uniform fractions; larger α → more medium-sized cuts).

---

## Classification-specific (not embedding geometry)

### `--label_smoothing`

**What it does:** Uses **label smoothing** on the **Thalamus cross-entropy** target (when not in ordinal mode). Instead of a hard “100% this class” target, the true class gets **slightly less than 100%** and other classes share the remainder.

**Plain English:** Stops the model from being overconfident on noisy discrete labels. It is **not** the same as randomly changing input vectors; it softens what “correct class” means in the loss.

---

### `--thal_ordinal` (interaction)

When training treats Thalamus as **ordinal**, **mixup** and **gene CutMix are disabled** because the loss and targets are structured differently. Dataset `--aug_*` can still apply unless you use **`--no_online_augment`**.

---

## Deprecated / related

**Offline `_augScaling` pickles** (via **`--data_suffix`** pointing at old augmented files) are **deprecated** in favor of the **base** pickle plus **online** `--aug_*` above. The training script may warn if you load those; prefer scaling jitter and the other flags here.

---

## Quick reference table

| Flag | Where it runs | One-line summary |
|------|----------------|------------------|
| `--no_online_augment` | Globally for static aug | Turns off all `--aug_*` dataset augmentations. |
| `--aug_scale_jitter` | Per sample, train | One random scale for the whole embedding. |
| `--aug_per_gene_scale_jitter` | Per sample, train (gene matrix) | Independent scale per gene row. |
| `--aug_gene_dropout` | Per sample, train (3D) | Randomly zeros whole genes. |
| `--aug_emb_dim_dropout` | Per sample, train | Randomly zeros whole embedding dimensions (shared across genes in 3D). |
| `--aug_feature_dropout` | Per sample, train (pooled 1D) | Randomly zeros pooled features. |
| `--aug_gene_block_cutout_max` | Per sample, train (3D) | Max genes in a contiguous zeroed block (0 = off). |
| `--aug_gene_block_cutout_prob` | Per sample, train | Chance to apply that block cutout when max &gt; 0. |
| `--aug_noise_std` | Per sample, train | Additive Gaussian noise to all entries. |
| `--mixup_alpha` | Per batch, train | Beta parameter for random mix *between two patients* (0 = off). |
| `--gene_cutmix_prob` | Per batch, train | Chance to splice a gene block from another patient (3D; off if ordinal). |
| `--gene_cutmix_alpha` | Per batch, train | Beta parameter for how *long* that splice tends to be. |
| `--label_smoothing` | Loss | Softens Thalamus class targets in cross-entropy. |

For **exact mathematical behavior and ordering**, see **`src/train_model.py`** (`EmbeddingAugmentConfig`, `augment_embedding_tensor`, and `train_epoch`).
