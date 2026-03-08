# GREmLN Astrocytes embedding visualization

Interactive 2D scatter plot of GREmLN cell embeddings (Astrocytes only), linked to atlas metadata by cell ID (`cell_annotation`). Embedding row IDs like `6289-MW-0031_bin2` are linked to metadata row `cell_annotation == 6289-MW-0031`.

## Setup

From the project root:

```bash
pip install -r embedding_viz/requirements.txt
```

## Run

From the project root:

```bash
python -m embedding_viz.app
```

Then open http://127.0.0.1:5000 in a browser.

- **Region**: choose which Astrocytes region (EC, ITG, MTG, PFC, V1, V2) to visualize.
- **Color by**: color points by a metadata column (e.g. Path..Group., Thal, SampleName) when metadata is available.
- Points are 2D PCA of the 512-d cell embeddings. Hover for cell ID and metadata.

## Data

- Embeddings: `GREmLN/embeddings/*_cell_embeddings.tsv` (files without "Microglia" in the name).
- Metadata: `data/AtlasData/2025-11-16_Astrocytes_metadata.xlsx` (one sheet per region). If the file is missing, the plot still works but without metadata columns for coloring or tooltips.
- Cache: reduced coordinates are cached under `.cache/embedding_viz/` per region for faster reloads.

## Precompute cache for all regions

To build the cache for every region in one go (so the app loads instantly when switching regions):

```bash
python -m embedding_viz.precompute_cache
```

Use `--force` to recompute even when cache exists:

```bash
python -m embedding_viz.precompute_cache --force
```
