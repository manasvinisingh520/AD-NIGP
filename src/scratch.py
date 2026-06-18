# load atlas anndata object and print sample of adata.obs
import anndata as ad
import matplotlib.pyplot as plt

microglia_adata = ad.read_h5ad('data/ROSMAP_Data/Microglia.h5ad')
astrocytes_adata = ad.read_h5ad('data/ROSMAP_Data/Astrocytes.h5ad')

def donor_level_counts(adata, column, id_col="individualID"):
    per_donor = adata.obs.drop_duplicates(id_col)
    return per_donor[column].value_counts()

"""all_columns = input("Enter all columns to print: ")
all_columns = [c.strip() for c in all_columns.split(',') if c.strip()]

for column in all_columns:
    print(f"{column} (Microglia, per individualID):\n")
    print(donor_level_counts(microglia_adata, column))
    print()

print("=" * 100)

for column in all_columns:
    print(f"{column} (Astrocytes, per individualID):\n")
    print(donor_level_counts(astrocytes_adata, column))
    print()"""

def patient_cells_heatmap(adata, id_col="individualID", title=None):
    """Heatmap of cell counts per donor (one column = one individualID)."""
    if id_col not in adata.obs.columns:
        raise ValueError(f"Missing {id_col!r}. Available: {list(adata.obs.columns)}")

    patient_cells = adata.obs.groupby(id_col).size().sort_values(ascending=False)
    n = len(patient_cells)
    fig_w = min(max(8, n * 0.08), 40)

    fig, ax = plt.subplots(figsize=(fig_w, 2))
    im = ax.imshow(patient_cells.values.reshape(1, -1), aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="cells per donor")
    ax.set_yticks([0])
    ax.set_yticklabels(["cell count"])
    ax.set_xticks(range(n))
    ax.set_xticklabels(patient_cells.index, rotation=90, fontsize=max(4, 8 - n // 80))
    ax.set_title(title or f"Cells per {id_col} ({adata.n_obs:,} cells, {n} donors)")
    fig.tight_layout()
    plt.show()

patient_cells_heatmap(microglia_adata, title="Microglia")
patient_cells_heatmap(astrocytes_adata, title="Astrocytes")
