import anndata as ad

atac_data = ad.read_h5ad("data/SEAAD_ATAC_Data/SEAAD_MTG_ATACseq_final-nuclei.2024-12-06.h5ad")
print(atac_data.X[:5,:5])